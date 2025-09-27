import torch
import torch.nn as nn

class EdgeDecoder(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, z_src, z_dst, z_label, return_logits: bool = False):
        x = torch.cat([z_src, z_dst, z_label], dim=-1)
        logit = self.mlp(x)
        if return_logits:
            return logit  # senza sigmoid
        return torch.sigmoid(logit)
