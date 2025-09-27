import os
import json
import numpy as np
import pandas as pd


def summary_anomalies(
    anomalies_path="outputs/anomalies.jsonl",
    out_csv="outputs/anomaly_summary.csv",
):
    events = []
    with open(anomalies_path, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))

    if not events:
        print("[ERR] anomalies.jsonl vuoto.")
        return

    scores = np.array([float(ev["anomaly_score"]) for ev in events])
    n_total = len(scores)

    # calcola conteggi sopra varie soglie
    thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
    rows = []
    for thr in thresholds:
        n_anom = int(np.sum(scores >= thr))
        perc = n_anom / n_total * 100
        rows.append({"threshold": thr, "n_anomalies": n_anom, "perc": perc})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Analisi completata su {n_total} eventi.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    summary_anomalies()
