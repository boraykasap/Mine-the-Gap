import os
import json
import pandas as pd


def analyze_top_anomalies(
    anomalies_path="outputs/anomalies.jsonl",
    out_csv="outputs/top_anomalies.csv",
    top_k=100,
):
    # carica tutti gli eventi con score
    events = []
    with open(anomalies_path, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))

    if not events:
        print("[ERR] anomalies.jsonl vuoto, nulla da analizzare.")
        return

    # ordina per anomaly_score discendente
    events = sorted(events, key=lambda x: -x.get("anomaly_score", 0.0))

    # seleziona top-K
    top_events = events[:top_k]

    # converti in DataFrame per esportazione
    df = pd.DataFrame(top_events)

    # colonne utili (riordino per chiarezza)
    cols = ["event_id", "timestamp", "src", "dst", "label", "event_type", "p_model", "anomaly_score"]
    df = df[cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Salvati top-{top_k} eventi sospetti in {out_csv}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    analyze_top_anomalies()
