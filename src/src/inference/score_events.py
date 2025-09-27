import os
import json
import argparse
import torch
import pandas as pd
from src.models.tgn import TGN
from src.models.decoder import EdgeDecoder


def score():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input/edge_events.csv",
                        help="CSV di input (clean o noisy)")
    parser.add_argument("--output", type=str, default="outputs/anomalies.jsonl",
                        help="File JSONL di output con gli anomaly score")
    parser.add_argument("--ckpt", type=str, default="artifacts/tgn_checkpoint.pt",
                        help="Checkpoint del modello")
    args = parser.parse_args()

    # carica checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")  # se 'weights_only' dà errore, toglilo
    node2id = ckpt["node2id"]
    label2id = ckpt["label2id"]
    emb_dim = ckpt.get("emb_dim", 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgn = TGN(num_nodes=len(node2id), num_labels=len(label2id), emb_dim=emb_dim).to(device)
    dec = EdgeDecoder(emb_dim=emb_dim).to(device)

    # carica pesi modello
    if "tgn" in ckpt and "decoder" in ckpt:
        tgn.load_state_dict(ckpt["tgn"])
        dec.load_state_dict(ckpt["decoder"])
    elif "model_state_dict" in ckpt:
        tgn.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError("Checkpoint non contiene né 'tgn'/'decoder' né 'model_state_dict'")

    tgn.eval(); dec.eval()
    tgn.reset_state()

    # carica eventuale temperatura (se esiste calibration.json)
    T_opt = 1.0
    calib_path = "artifacts/calibration.json"
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            T_opt = json.load(f).get("temperature", 1.0)
        print(f"[INFO] Calibrazione caricata: T={T_opt:.4f}")
    else:
        print("[INFO] Nessuna calibrazione trovata, uso T=1.0")

    # carica eventi
    df = pd.read_csv(args.input)
    print(f"[DIAG] Tot eventi input: {len(df)}")

    out = []
    miss_nodes, miss_labels = 0, 0

    for _, r in df.iterrows():
        src = node2id.get(str(r.src))
        dst = node2id.get(str(r.dst))
        lab = label2id.get(str(r.label))

        if src is None or dst is None or lab is None:
            if src is None or dst is None:
                miss_nodes += 1
            if lab is None:
                miss_labels += 1
            score = 1.0
            p = 0.0
        else:
            s = torch.tensor([src], dtype=torch.long, device=device)
            d = torch.tensor([dst], dtype=torch.long, device=device)
            l = torch.tensor([lab], dtype=torch.long, device=device)
            t = torch.tensor([int(r.timestamp)], dtype=torch.long, device=device)

            with torch.no_grad():
                # ottieni logit
                logit = dec(z_s := tgn.embed_pair(s, d, l)[0],
                            z_d := tgn.embed_pair(s, d, l)[1],
                            z_l := tgn.embed_pair(s, d, l)[2],
                            return_logits=True).item()
                # calibra
                logit_scaled = logit / T_opt
                p = torch.sigmoid(torch.tensor(logit_scaled)).item()

            if r.event_type == "add":
                score = 1.0 - p
            elif r.event_type == "remove":
                score = p
            else:
                score = 1.0 - p

            tgn.update_with_events(s, d, t, l)

        out.append({
            "event_id": f"{int(r.timestamp)}_{r.src}_{r.dst}_{r.event_type}",
            "timestamp": int(r.timestamp),
            "src": str(r.src),
            "dst": str(r.dst),
            "label": str(r.label),
            "event_type": str(r.event_type),
            "p_model": float(p),
            "anomaly_score": float(score)
        })

    print(f"[DIAG] Eventi processati: {len(out)}")
    print(f"[DIAG] Mismatch residui: new_nodes={miss_nodes}, new_labels={miss_labels}")
    print(f"[DIAG] Mapping training: {len(node2id)} nodi, {len(label2id)} label")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"[DONE] Salvato {args.output}")


if __name__ == "__main__":
    score()
