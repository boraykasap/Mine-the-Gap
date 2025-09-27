import pandas as pd
from typing import Tuple, Dict, List

REQUIRED_COLS = ["src", "dst", "label", "timestamp", "event_type"]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["label"] = df["label"].astype(str)
    df["event_type"] = df["event_type"].astype(str).str.lower()
    return df

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = _normalize_df(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def filter_add_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["event_type"] == "add"].reset_index(drop=True)

def build_maps_union(clean_df: pd.DataFrame, noisy_df: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,int]]:
    """Mappe su UNIONE di clean+noisy (evita mismatch in inference)."""
    clean_df = _normalize_df(clean_df)
    noisy_df = _normalize_df(noisy_df)
    nodes = pd.concat([clean_df["src"], clean_df["dst"], noisy_df["src"], noisy_df["dst"]]).unique().tolist()
    labels = pd.concat([clean_df["label"], noisy_df["label"]]).unique().tolist()
    node2id = {n: i for i, n in enumerate(nodes)}
    label2id = {l: i for i, l in enumerate(labels)}
    return node2id, label2id
