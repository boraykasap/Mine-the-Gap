import pandas as pd
import matplotlib.pyplot as plt

# Load scored events
df = pd.read_json("outputs/anomalies.jsonl", lines=True)

# Histogram of anomaly scores
plt.hist(df["anomaly_score"], bins=50, color="steelblue")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Distribution of Anomaly Scores")
plt.savefig("outputs/charts/score_distribution.png")

# Anomalies over time
threshold = 0.9
ts = df[df.anomaly_score >= threshold].groupby("timestamp").size()
ts.plot(kind="line", color="red")
plt.xlabel("Timestamp")
plt.ylabel("# of anomalies >= 0.9")
plt.title("Anomalies Over Time")
plt.savefig("outputs/charts/anomalies_over_time.png")