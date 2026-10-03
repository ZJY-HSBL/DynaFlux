from __future__ import annotations

import math

import torch
from torch import nn


class TopologyAttentionGraph(nn.Module):
    """Topology-masked multi-head graph attention that returns a local association matrix."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, topology: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,D], topology: [N,N]
        node = x.mean(dim=1)
        b, n, _ = node.shape
        q = self.q(node).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(node).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        topo = topology.to(dtype=scores.dtype, device=scores.device)
        mask = topo > 0
        # Preserve distance affinity as a small additive prior.
        prior = torch.log(topo.clamp_min(1e-6))[None, None, :, :]
        scores = scores + prior
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return attn.mean(dim=1)
