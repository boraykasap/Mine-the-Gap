import os
import json
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from src.models.tgn import TGN
from src.models.decoder import EdgeDecoder

"""
Calibrazione post-hoc del modello TGN tramite Temperature Scaling.
- Usa il dataset clean (edge_events_clean.csv).
- Ottimizza un parametro T per calibrare le probabilità.
- Salva T in artifacts/calibration.json.
"""

def calibrate():
    ckpt_path = "artifacts/tgn_checkpoint.pt"
    input_file = "input/edge_events_clean.csv"
    calib_out = "artifacts/calibration.json"

    # Carica modello e checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    node2id = ckpt["node2id"]
    label2id = ckpt["label2id"]
    emb_dim = ckpt.get("emb_dim", 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgn = TGN(num_nodes=len(node2id), num_labels=len(label2id), emb_dim=emb_dim).to(device)
    dec = EdgeDecoder(emb_dim=emb_dim).to(device)
    tgn.load_state_dict(ckpt["tgn"])
    dec.load_state_dict(ckpt["decoder"])
    tgn.eval(); dec.eval()
    tgn.reset_state()

    # Carica dataset clean
    df = pd.read_csv(input_file)
    n = len(df)
    split_idx = int(n * 0.9)
    val_df = df.iloc[split_idx:]  # ultimo 10% come validation

    logits, labels = [], []

    for _, r in val_df.iterrows():
        src = node2id.get(str(r.src))
        dst = node2id.get(str(r.dst))
        lab = label2id.get(str(r.label))
        if src is None or dst is None or lab is None:
            continue

        s = torch.tensor([src], dtype=torch.long, device=device)
        d = torch.tensor([dst], dtype=torch.long, device=device)
        l = torch.tensor([lab], dtype=torch.long, device=device)
        t = torch.tensor([int(r.timestamp)], dtype=torch.long, device=device)

        with torch.no_grad():
            z_s, z_d, z_l = tgn.embed_pair(s, d, l)
            logit = dec(z_s, z_d, z_l, return_logits=True).item()

        logits.append(logit)
        labels.append(1 if r.event_type == "add" else 0)
        tgn.update_with_events(s, d, t, l)

    logits = torch.tensor(logits, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    # Parametro Temperature
    T = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / T
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_opt = T.item()

    print(f"[DONE] Temperature ottimale trovata: T={T_opt:.4f}")

    os.makedirs(os.path.dirname(calib_out), exist_ok=True)
    with open(calib_out, "w") as f:
        json.dump({"temperature": T_opt}, f)

    print(f"[DONE] Salvato {calib_out}")


if __name__ == "__main__":
    calibrate()
