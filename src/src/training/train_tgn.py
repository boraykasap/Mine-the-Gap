import os
import json # <-- Aggiunto
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocess import load_events, filter_add_events, build_maps_union
from src.data.dataset import EventDataset
from src.data.sampler import NegativeSampler
from src.models.tgn import TGN
from src.models.decoder import EdgeDecoder

BATCH_SIZE = 2048
NEG_K = 5
EMB_DIM = 64
EPOCHS = 50
LR = 1e-3

def collate_with_neg(batch, neg_sampler: NegativeSampler):
    batch = [b for b in batch if b["src"] is not None and b["dst"] is not None and b["label"] is not None]
    if len(batch) == 0:
        return None
    p_src = torch.tensor([b["src"] for b in batch], dtype=torch.long)
    p_dst = torch.tensor([b["dst"] for b in batch], dtype=torch.long)
    p_label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    p_t = torch.tensor([b["timestamp"] for b in batch], dtype=torch.long)
    p_y = torch.ones((len(batch), 1), dtype=torch.float32)

    neg_dst = neg_sampler.sample_for_src_batch(p_src.tolist())
    n_src = torch.repeat_interleave(p_src, repeats=neg_sampler.k)
    n_dst = torch.tensor(neg_dst, dtype=torch.long)
    n_label = torch.repeat_interleave(p_label, repeats=neg_sampler.k)
    n_t = torch.repeat_interleave(p_t, repeats=neg_sampler.k)
    n_y = torch.zeros((len(n_dst), 1), dtype=torch.float32)

    all_src = torch.cat([p_src, n_src], dim=0)
    all_dst = torch.cat([p_dst, n_dst], dim=0)
    all_label = torch.cat([p_label, n_label], dim=0)
    all_y = torch.cat([p_y, n_y], dim=0)
    return p_src, p_dst, p_label, p_t, all_src, all_dst, all_label, all_y

def train():
    # 1) Carica dati
    clean_df = load_events("input/edge_events_clean.csv")
    noisy_df = load_events("input/edge_events.csv")
    # 2) Mappe su UNIONE
    node2id, label2id = build_maps_union(clean_df, noisy_df)
    # 3) Training solo su 'add' del clean
    train_df = filter_add_events(clean_df)
    dataset = EventDataset(train_df, node2id, label2id)
    neg_sampler = NegativeSampler(num_nodes=len(node2id), k=NEG_K, seed=42)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                        collate_fn=lambda b: collate_with_neg(b, neg_sampler))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgn = TGN(num_nodes=len(node2id), num_labels=len(label2id), emb_dim=EMB_DIM).to(device)
    dec = EdgeDecoder(emb_dim=EMB_DIM).to(device)

    opt = torch.optim.Adam(list(tgn.parameters()) + list(dec.parameters()), lr=LR)
    bce = nn.BCELoss()

    tgn.train(); dec.train()
    tgn.reset_state()

    # --- NUOVA AGGIUNTA: History della loss ---
    loss_history = []
    # -----------------------------------------

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for pack in loader:
            if pack is None:
                continue
            p_src, p_dst, p_label, p_t, all_src, all_dst, all_label, all_y = pack
            p_src, p_dst = p_src.to(device), p_dst.to(device)
            p_label, p_t = p_label.to(device), p_t.to(device)
            all_src, all_dst = all_src.to(device), all_dst.to(device)
            all_label, all_y = all_label.to(device), all_y.to(device)

            tgn.detach_memory()

            z_s, z_d, z_l = tgn.embed_pair(all_src, all_dst, all_label)
            preds = dec(z_s, z_d, z_l)
            loss = bce(preds, all_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())

            tgn.update_with_events(p_src, p_dst, p_t, p_label)

        print(f"[Epoch {epoch+1}] loss={total_loss:.6f}")
        # --- NUOVA AGGIUNTA: Salva loss dell'epoca ---
        loss_history.append({"epoch": epoch + 1, "loss": total_loss})
        # -------------------------------------------

    # --- NUOVA AGGIUNTA: Salva il log della loss su file ---
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/training_log.json", "w") as f:
        json.dump(loss_history, f)
    print("[DONE] Log di training salvato in outputs/training_log.json")
    # -----------------------------------------------------

    os.makedirs("artifacts", exist_ok=True)
    torch.save({
        "tgn": tgn.state_dict(),
        "decoder": dec.state_dict(),
        "node2id": node2id,
        "label2id": label2id,
        "emb_dim": EMB_DIM,
        "neg_k": NEG_K
    }, "artifacts/tgn_checkpoint.pt")

if __name__ == "__main__":
    train()