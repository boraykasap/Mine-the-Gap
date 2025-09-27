import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni grafiche per un look più professionale
sns.set_theme(style="whitegrid")
REPORT_DIR = "outputs/report"
ANOMALIES_PATH = "outputs/anomalies.jsonl"
CLEAN_ANOMALIES_PATH = "outputs/anomalies_clean.jsonl"
COMPARE_PATH = "outputs/compare_clean_noisy.csv"
TOP_NODES_PATH = "outputs/derived/nodes.json"
TRAINING_LOG_PATH = "outputs/training_log.json"  # <-- Aggiunto


def plot_training_loss():
    """Trama la curva di apprendimento (loss vs. epoche)."""
    if not os.path.exists(TRAINING_LOG_PATH):
        print(f"[WARN] File {TRAINING_LOG_PATH} non trovato. Salto grafico della loss.")
        return

    df_loss = pd.read_json(TRAINING_LOG_PATH)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_loss, x='epoch', y='loss', marker='o', color='purple')
    plt.title('Andamento della Loss di Training per Epoca', fontsize=16)
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Loss Totale', fontsize=12)
    plt.xticks(df_loss['epoch'][::5])  # Mostra un tick ogni 5 epoche per leggibilità
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "0_training_loss.png"))
    print(f"[OK] Grafico salvato: 0_training_loss.png")


def plot_score_distribution(df: pd.DataFrame):
    """Trama la distribuzione degli anomaly score."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['anomaly_score'], bins=50, kde=True, color="steelblue")
    plt.title('Distribuzione degli Anomaly Score (Dataset Noisy)', fontsize=16)
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Numero di Eventi', fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "1_score_distribution.png"))
    print(f"[OK] Grafico salvato: 1_score_distribution.png")


def plot_anomalies_over_time(df: pd.DataFrame, threshold=0.9):
    """Trama il numero di anomalie e lo score medio nel tempo."""
    df_time = df.copy()
    df_time['is_anomaly'] = df_time['anomaly_score'] >= threshold

    ts_agg = df_time.groupby('timestamp').agg(
        n_anomalies=('is_anomaly', 'sum'),
        mean_score=('anomaly_score', 'mean')
    ).reset_index()

    ts_agg['mean_score_smooth'] = ts_agg['mean_score'].rolling(window=50, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(14, 7))

    sns.lineplot(data=ts_agg, x='timestamp', y='n_anomalies', ax=ax1, color='lightcoral',
                 label=f'# Anomalie (score >= {threshold})', alpha=0.6)
    ax1.set_xlabel('Timestamp', fontsize=12)
    ax1.set_ylabel(f'# Anomalie (score >= {threshold})', color='lightcoral', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='lightcoral')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    sns.lineplot(data=ts_agg, x='timestamp', y='mean_score_smooth', ax=ax2, color='steelblue',
                 label='Score Medio (smussato)')
    ax2.set_ylabel('Anomaly Score Medio (smussato)', color='steelblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(0, 1)

    plt.title('Andamento delle Anomalie nel Tempo', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "2_anomalies_over_time.png"))
    print(f"[OK] Grafico salvato: 2_anomalies_over_time.png")


def plot_clean_vs_noisy_distribution():
    """Confronta le distribuzioni degli score tra dataset clean e noisy."""
    if not os.path.exists(CLEAN_ANOMALIES_PATH):
        print(f"[WARN] File {CLEAN_ANOMALIES_PATH} non trovato. Salto grafico di confronto.")
        return

    df_clean = pd.read_json(CLEAN_ANOMALIES_PATH, lines=True)
    df_noisy = pd.read_json(ANOMALIES_PATH, lines=True)

    plt.figure(figsize=(12, 7))
    sns.kdeplot(df_clean['anomaly_score'], label='Clean Dataset', color='green', fill=True, alpha=0.5)
    sns.kdeplot(df_noisy['anomaly_score'], label='Noisy Dataset', color='red', fill=True, alpha=0.5)
    plt.title('Confronto Distribuzione Anomaly Score: Clean vs. Noisy', fontsize=16)
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Densità', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "3_clean_vs_noisy_dist.png"))
    print(f"[OK] Grafico salvato: 3_clean_vs_noisy_dist.png")


def plot_top_anomalous_nodes(threshold=0.9):
    """Mostra i nodi più frequentemente coinvolti in eventi anomali."""
    if not os.path.exists(TOP_NODES_PATH):
        print(f"[WARN] File {TOP_NODES_PATH} non trovato. Esegui prima analyze_graph.py. Salto grafico.")
        return

    with open(TOP_NODES_PATH, 'r') as f:
        nodes_data = json.load(f)

    df_nodes = pd.DataFrame(nodes_data)
    df_top_nodes = df_nodes.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='node', data=df_top_nodes, palette='viridis')
    plt.title(f'Top 20 Nodi Coinvolti in Eventi Anomali (score >= {threshold})', fontsize=16)
    plt.xlabel('Numero di Eventi Anomali', fontsize=12)
    plt.ylabel('ID Nodo', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "4_top_anomalous_nodes.png"))
    print(f"[OK] Grafico salvato: 4_top_anomalous_nodes.png")


def generate_report():
    """Funzione principale per generare tutti i grafici e i report."""
    print("--- Inizio Generazione Report ---")
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(ANOMALIES_PATH):
        print(f"[ERR] File {ANOMALIES_PATH} non trovato. Esegui prima score_events.py.")
        return

    df_anomalies = pd.read_json(ANOMALIES_PATH, lines=True)

    # Genera grafici
    plot_training_loss()  # <-- Aggiunto
    plot_score_distribution(df_anomalies)
    plot_anomalies_over_time(df_anomalies)
    plot_clean_vs_noisy_distribution()
    plot_top_anomalous_nodes()

    if os.path.exists("outputs/metrics.json"):
        with open("outputs/metrics.json", "r") as f:
            metrics = json.load(f)
        print("\n--- Riepilogo Metriche di Valutazione ---")
        print(json.dumps(metrics, indent=2))
        print("------------------------------------")

    print("\n--- Report Generato con Successo ---")
    print(f"Tutti i grafici sono stati salvati in: {REPORT_DIR}")


if __name__ == "__main__":
    generate_report()