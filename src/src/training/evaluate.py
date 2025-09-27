import json
import math
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.preprocess import load_events
from src.data.sampler import NegativeSampler
from src.models.tgn import TGN
from src.models.decoder import EdgeDecoder


# ---------- Utility base ----------

def precision_at_k(y_true, y_score, k):
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    idx = np.argsort(-np.asarray(y_score))[:k]
    return float(np.sum(np.asarray(y_true)[idx]) / k)

def recall_at_k(y_true, y_score, k):
    if len(y_true) == 0:
        return 0.0
    tot_pos = int(np.sum(y_true))
    if tot_pos == 0:
        return 0.0
    k = min(k, len(y_true))
    idx = np.argsort(-np.asarray(y_score))[:k]
    tp = int(np.sum(np.asarray(y_true)[idx]))
    return float(tp / tot_pos)

def build_id_strict(df_row):
    return f"{int(df_row.timestamp)}_{df_row.src}_{df_row.dst}_{df_row.event_type}"

def build_id_relaxed(df_row):
    return f"{df_row.src}_{df_row.dst}_{df_row.label}"

def bin_timestamps(ts_series, n_bins=50):
    t_min = int(ts_series.min())
    t_max = int(ts_series.max())
    if t_max == t_min:
        # tutti uguali: un solo bucket
        return pd.Series([0] * len(ts_series)), 1, [(t_min, t_max)]
    width = max(1, math.ceil((t_max - t_min + 1) / n_bins))
    bins = ((ts_series - t_min) // width).astype(int)
    nb = int(bins.max()) + 1
    ranges = [(t_min + i * width, t_min + (i + 1) * width - 1) for i in range(nb)]
    return bins, nb, ranges


# ---------- 1) Internal temporal link prediction su CLEAN ----------

def eval_internal_lp_on_clean(ckpt_path="artifacts/tgn_checkpoint.pt",
                              clean_path="input/edge_events_clean.csv",
                              neg_k=5, emb_dim=64, device=None):
    """
    Split temporale: usiamo ultimo 10% del CLEAN come validation.
    Per ogni evento di validazione:
      - calcoliamo p(edge) prima dell'update;
      - generiamo K negativi con stesso src e timestamp;
    Calcoliamo ROC-AUC / AP / P@K / R@K sulla validation.
    """
    clean = load_events(clean_path)
    if len(clean) < 10:
        print("[LP] Clean troppo piccolo per una validation sensata.")
        return None

    # carica modello e mappe
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    node2id, label2id = ckpt["node2id"], ckpt["label2id"]
    emb_dim = ckpt.get("emb_dim", emb_dim)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tgn = TGN(num_nodes=len(node2id), num_labels=len(label2id), emb_dim=emb_dim).to(device)
    dec = EdgeDecoder(emb_dim=emb_dim).to(device)
    tgn.load_state_dict(ckpt["tgn"]); dec.load_state_dict(ckpt["decoder"])
    tgn.eval(); dec.eval()
    tgn.reset_state()

    # scorriamo tutto il CLEAN: i primi 90% servono per "riempire" la memoria,
    # il restante 10% è la validation su cui misuriamo LP.
    n = len(clean)
    split_at = int(n * 0.9)
    warmup_df = clean.iloc[:split_at]
    val_df = clean.iloc[split_at:]

    # warmup: aggiorna la memoria con gli eventi iniziali (solo se sono mappabili)
    with torch.no_grad():
        for _, r in warmup_df.iterrows():
            s = node2id.get(str(r.src), None)
            d = node2id.get(str(r.dst), None)
            l = label2id.get(str(r.label), None)
            if s is None or d is None or l is None:
                continue
            s = torch.tensor([s], dtype=torch.long, device=device)
            d = torch.tensor([d], dtype=torch.long, device=device)
            l = torch.tensor([l], dtype=torch.long, device=device)
            t = torch.tensor([int(r.timestamp)], dtype=torch.long, device=device)
            # qui non misuriamo, aggiorniamo solo lo stato
            tgn.update_with_events(s, d, t, l)

    # validation: misura p(edge) prima dell'update + negativi
    sampler = NegativeSampler(num_nodes=len(node2id), k=neg_k, seed=123)
    y_true, y_score = [], []

    for _, r in val_df.iterrows():
        s_id = node2id.get(str(r.src), None)
        d_id = node2id.get(str(r.dst), None)
        l_id = label2id.get(str(r.label), None)
        if s_id is None or d_id is None or l_id is None:
            continue

        s = torch.tensor([s_id], dtype=torch.long, device=device)
        d = torch.tensor([d_id], dtype=torch.long, device=device)
        l = torch.tensor([l_id], dtype=torch.long, device=device)
        t = torch.tensor([int(r.timestamp)], dtype=torch.long, device=device)

        with torch.no_grad():
            z_s, z_d, z_l = tgn.embed_pair(s, d, l)
            p_pos = float(dec(z_s, z_d, z_l).item())

        y_true.append(1)
        y_score.append(p_pos)

        # negativi per lo stesso src
        neg_dsts = sampler.sample_for_src_batch([int(s_id)])
        for nd in neg_dsts:
            nd_t = t  # stesso tempo
            nd_id = torch.tensor([nd], dtype=torch.long, device=device)
            with torch.no_grad():
                z_s, z_d, z_l = tgn.embed_pair(s, nd_id, l)
                p_neg = float(dec(z_s, z_d, z_l).item())
            y_true.append(0)
            y_score.append(p_neg)

        # aggiorna memoria SOLO dopo aver misurato
        with torch.no_grad():
            tgn.update_with_events(s, d, t, l)

    metrics = {}
    if len(set(y_true)) >= 2:
        metrics["lp_roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["lp_average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["lp_roc_auc"] = None
        metrics["lp_average_precision"] = None

    for k in [10, 50, 100]:
        metrics[f"lp_precision_at_{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"lp_recall_at_{k}"] = recall_at_k(y_true, y_score, k)

    print("[LP] Internal link prediction (validation su clean, ultimo 10%)")
    print(f"   ROC-AUC={metrics['lp_roc_auc']}, AP={metrics['lp_average_precision']}")
    for k in [10, 50, 100]:
        print(f"   P@{k}={metrics[f'lp_precision_at_{k}']:.4f}  R@{k}={metrics[f'lp_recall_at_{k}']:.4f}")

    return metrics


# ---------- 2) Diff diagnostico: strict e relaxed ----------

def diff_diagnostics(clean_path="input/edge_events_clean.csv",
                     noisy_path="input/edge_events.csv"):
    clean = load_events(clean_path)
    noisy = load_events(noisy_path)

    strict_clean = set(build_id_strict(r) for _, r in clean.iterrows())
    strict_noisy = set(build_id_strict(r) for _, r in noisy.iterrows())

    relaxed_clean = set(build_id_relaxed(r) for _, r in clean.iterrows())
    relaxed_noisy = set(build_id_relaxed(r) for _, r in noisy.iterrows())

    diag = {
        "strict_common": len(strict_clean & strict_noisy),
        "strict_only_clean": len(strict_clean - strict_noisy),
        "strict_only_noisy": len(strict_noisy - strict_clean),
        "relaxed_common": len(relaxed_clean & relaxed_noisy),
        "relaxed_only_clean": len(relaxed_clean - relaxed_noisy),
        "relaxed_only_noisy": len(relaxed_noisy - relaxed_clean),
        "n_clean": len(clean),
        "n_noisy": len(noisy),
    }

    print("[DIFF] strict: common={}, clean_only={}, noisy_only={}".format(
        diag["strict_common"], diag["strict_only_clean"], diag["strict_only_noisy"]))
    print("[DIFF] relaxed: common={}, clean_only={}, noisy_only={}".format(
        diag["relaxed_common"], diag["relaxed_only_clean"], diag["relaxed_only_noisy"]))

    return diag


# ---------- 3) Diff finestrato (temporal windowed) ----------

def windowed_diff_metrics(anomalies_path="outputs/anomalies.jsonl",
                          clean_path="input/edge_events_clean.csv",
                          noisy_path="input/edge_events.csv",
                          n_bins=50):
    """
    Discretizza il tempo in n_bins, confronta la presenza di (src,dst,label) per finestra.
    Se (triplet,bin) è presente nel noisy ma non nel clean (o viceversa) => anomalia temporale.
    Poi mappa ogni evento predetto (da anomalies.jsonl) al suo bin e calcola P@K/R@K,
    e AUC/AP se bilanciata.
    """
    clean = load_events(clean_path)
    noisy = load_events(noisy_path)

    # binning
    all_ts = pd.concat([clean["timestamp"], noisy["timestamp"]], axis=0).reset_index(drop=True)
    bins, nb, ranges = bin_timestamps(all_ts, n_bins=n_bins)
    # Applica lo stesso binning separatamente
    t_min = int(all_ts.min())
    width = max(1, math.ceil((int(all_ts.max()) - t_min + 1) / n_bins))
    def to_bin(ts): return (ts - t_min) // width

    clean["bin"] = clean["timestamp"].apply(to_bin)
    noisy["bin"] = noisy["timestamp"].apply(to_bin)

    def rows_to_set(df):
        return set((row.bin, row.src, row.dst, row.label) for _, row in df.iterrows())

    set_clean = rows_to_set(clean)
    set_noisy = rows_to_set(noisy)

    # pseudo-GT finestrato
    w_anom = (set_noisy - set_clean) | (set_clean - set_noisy)

    # carica predizioni
    events, scores = [], []
    with open(anomalies_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            events.append(d)
            scores.append(float(d["anomaly_score"]))

    y_true = []
    for ev in events:
        # parse event_id: "{timestamp}_{src}_{dst}_{event_type}"
        parts = ev["event_id"].split("_")
        if len(parts) < 4:
            y_true.append(0)
            continue
        ts = int(parts[0]); src = parts[1]; dst = parts[2]
        # label la prendiamo da campo "label" nell'oggetto
        lab = ev.get("label", None)
        if lab is None:
            # se non c'è, non possiamo mappare correttamente
            y_true.append(0)
            continue
        b = int((ts - t_min) // width)
        key = (b, src, dst, lab)
        y_true.append(1 if key in w_anom else 0)

    metrics = {}
    print(f"[WIN] Tot eventi valutati: {len(y_true)}; Positivi={sum(y_true)}; Negativi={len(y_true)-sum(y_true)}")
    if len(set(y_true)) >= 2:
        metrics["win_roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["win_average_precision"] = float(average_precision_score(y_true, scores))
        print(f"[WIN] ROC-AUC={metrics['win_roc_auc']:.4f}  AP={metrics['win_average_precision']:.4f}")
    else:
        metrics["win_roc_auc"] = None
        metrics["win_average_precision"] = None
        print("[WIN] Classi non bilanciate → AUC/AP non calcolabili")

    for k in [10, 50, 100]:
        p = precision_at_k(y_true, scores, k)
        r = recall_at_k(y_true, scores, k)
        metrics[f"win_precision_at_{k}"] = p
        metrics[f"win_recall_at_{k}"] = r
        print(f"[WIN] P@{k}={p:.4f}  R@{k}={r:.4f}")

    return metrics


# ---------- Main orchestrator ----------

def evaluate():
    # 1) Internal LP su clean
    lp_metrics = eval_internal_lp_on_clean()

    # 2) Diff diagnostico
    diag = diff_diagnostics()

    # 3) Windowed diff
    win_metrics = windowed_diff_metrics()

    # 4) Salva tutto
    all_metrics = {"internal_lp": lp_metrics, "diff_diag": diag, "windowed_diff": win_metrics}
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("[DONE] metrics consolidate in outputs/metrics.json")


if __name__ == "__main__":
    evaluate()
