from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Transformer-style temporal encoder with calendar and node-adaptive embeddings."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        max_steps: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.flow_proj = nn.Linear(in_dim, hidden_dim)
        self.tod_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.dow_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_embed = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.step_embed = nn.Parameter(torch.empty(max_steps, hidden_dim))
        nn.init.xavier_normal_(self.node_embed)
        nn.init.xavier_normal_(self.step_embed)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,C], tod/dow: [B,T]
        b, t, n, _ = x.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {n}")
        if t > self.step_embed.shape[0]:
            raise ValueError(f"Sequence length {t} exceeds max_steps={self.step_embed.shape[0]}")

        h = self.flow_proj(x)
        h = h + self.tod_proj(tod[..., None])[:, :, None, :]
        h = h + self.dow_proj(dow[..., None])[:, :, None, :]
        h = h + self.node_embed[None, None, :, :]
        h = h + self.step_embed[:t][None, :, None, :]
        h = self.dropout(h)

        h = h.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden_dim)
        h = self.encoder(h)
        h = h.reshape(b, n, t, self.hidden_dim).permute(0, 2, 1, 3).contiguous()
        return h
