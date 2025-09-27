# src/postprocess/analyze_graph.py

import os
import json
import argparse
import pandas as pd

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/anomalies.jsonl",
                        help="Input file con gli anomaly score (jsonl)")
    parser.add_argument("--outdir", type=str, default="outputs/derived",
                        help="Cartella output per i file JSON derivati")
    parser.add_argument("--time_window", type=int, default=50,
                        help="Ampiezza finestra temporale per aggregare anomalie (in unità timestamp)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Soglia di default per considerare un evento anomalo")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load anomalies.jsonl ----
    events = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    df = pd.DataFrame(events)
    print(f"[INFO] Loaded {len(df)} events")

    # ---- Node ranking ----
    anomalous = df[df["anomaly_score"] >= args.threshold]
    node_counts = pd.concat([anomalous["src"], anomalous["dst"]]).value_counts().reset_index()
    node_counts.columns = ["node", "count"]
    node_counts = node_counts.to_dict(orient="records")

    with open(os.path.join(args.outdir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(node_counts, f, indent=2)
    print(f"[DONE] Saved node ranking → {args.outdir}/nodes.json")

    # ---- Time series ----
    bins = (df["timestamp"] // args.time_window) * args.time_window
    ts = df.groupby(bins)["anomaly_score"].apply(lambda x: (x >= args.threshold).sum()).reset_index()
    ts.columns = ["timestamp_bin", "n_anomalies"]
    ts = ts.to_dict(orient="records")

    with open(os.path.join(args.outdir, "timeseries.json"), "w", encoding="utf-8") as f:
        json.dump(ts, f, indent=2)
    print(f"[DONE] Saved anomaly time series → {args.outdir}/timeseries.json")

    # ---- Events list ----
    events_out = df.to_dict(orient="records")
    with open(os.path.join(args.outdir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events_out, f, indent=2)
    print(f"[DONE] Saved enriched events → {args.outdir}/events.json")


if __name__ == "__main__":
    analyze()
