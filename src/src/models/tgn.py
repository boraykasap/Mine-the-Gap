import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
# API ufficiale TGNMemory (raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)
# vedi docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.TGNMemory.html

class TGN(nn.Module):
    def __init__(self, num_nodes: int, num_labels: int, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.label_emb = nn.Embedding(num_labels, emb_dim)

        msg_module = IdentityMessage(raw_msg_dim=emb_dim, memory_dim=emb_dim, time_dim=emb_dim)
        aggr_module = LastAggregator()

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=emb_dim,
            memory_dim=emb_dim,
            time_dim=emb_dim,
            message_module=msg_module,
            aggregator_module=aggr_module,
        )

    # === Lookup embeddings (stato corrente, nessun update) ===
    def embed_pair(self, src: Tensor, dst: Tensor, label: Tensor):
        """
        Ottiene embedding correnti dei nodi e la raw message (label embedding).
        Usa lookup diretto su self.memory.memory per evitare side-effects.
        """
        z_src = self.memory.memory[src]  # [B, emb_dim]
        z_dst = self.memory.memory[dst]  # [B, emb_dim]
        z_lab = self.label_emb(label)    # [B, emb_dim]
        return z_src, z_dst, z_lab

    # === Update memoria con eventi osservati ===
    @torch.no_grad()
    def update_with_events(self, src: Tensor, dst: Tensor, t: Tensor, label: Tensor):
        raw_msg = self.label_emb(label).detach()
        self.memory.update_state(src=src, dst=dst, t=t, raw_msg=raw_msg)

    def reset_state(self):
        self.memory.reset_state()

    def detach_memory(self):
        self.memory.detach()
