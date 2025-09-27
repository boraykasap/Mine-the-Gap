import os
import json
import numpy as np
import pandas as pd


def load_scores(path: str):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    if not events:
        raise RuntimeError(f"[ERR] Nessun evento trovato in {path}")
    scores = np.array([float(ev["anomaly_score"]) for ev in events])
    return scores


def compare_clean_noisy(
    clean_path="outputs/anomalies_clean.jsonl",
    noisy_path="outputs/anomalies.jsonl",
    out_csv="outputs/compare_clean_noisy.csv",
):
    if not os.path.exists(clean_path):
        print(f"[WARN] File {clean_path} non trovato. Devi prima generarlo passando il file clean in score_events.py")
        return
    if not os.path.exists(noisy_path):
        print(f"[WARN] File {noisy_path} non trovato. Devi prima lanciare score_events.py sul file noisy.")
        return

    clean_scores = load_scores(clean_path)
    noisy_scores = load_scores(noisy_path)

    # statistiche base
    stats = {
        "clean_mean": float(np.mean(clean_scores)),
        "clean_std": float(np.std(clean_scores)),
        "clean_min": float(np.min(clean_scores)),
        "clean_max": float(np.max(clean_scores)),
        "noisy_mean": float(np.mean(noisy_scores)),
        "noisy_std": float(np.std(noisy_scores)),
        "noisy_min": float(np.min(noisy_scores)),
        "noisy_max": float(np.max(noisy_scores)),
    }

    print("[DIAG] Distribuzione anomaly score (Clean vs Noisy)")
    print(f" Clean → mean={stats['clean_mean']:.4f}, std={stats['clean_std']:.4f}, "
          f"min={stats['clean_min']:.4f}, max={stats['clean_max']:.4f}")
    print(f" Noisy → mean={stats['noisy_mean']:.4f}, std={stats['noisy_std']:.4f}, "
          f"min={stats['noisy_min']:.4f}, max={stats['noisy_max']:.4f}")

    # quantili
    quantiles = [0.5, 0.9, 0.95, 0.99]
    rows = []
    for q in quantiles:
        rows.append({
            "quantile": q,
            "clean_score": float(np.quantile(clean_scores, q)),
            "noisy_score": float(np.quantile(noisy_scores, q)),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("[DIAG] Quantili salvati in", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    compare_clean_noisy()
