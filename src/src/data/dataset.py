import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict

class EventDataset(Dataset):
    """
    Dataset sequenziale di eventi (ordinati per timestamp).
    Restituisce dizionari per facilitare la collate_fn.
    """
    def __init__(self, df: pd.DataFrame, node2id: Dict[str, int], label2id: Dict[str, int]):
        self.df = df
        self.node2id = node2id
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        # nota: assumiamo che i nodi/etichette esistano nelle mappe del training
        return {
            "src": self.node2id.get(r.src, None),
            "dst": self.node2id.get(r.dst, None),
            "label": self.label2id.get(r.label, None),
            "timestamp": int(r.timestamp),
            "event_type": r.event_type,
        }
