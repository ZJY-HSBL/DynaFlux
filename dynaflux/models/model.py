from __future__ import annotations

import torch
from torch import nn

from .adaptive_conv import AdaptiveGatedConv
from .dynamic_graph import MultiScaleDynamicGraph
from .temporal import TemporalEncoder


class DynaFluxBranch(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        temporal_heads: int,
        temporal_layers: int,
        graph_heads: int,
        num_clusters: int,
        num_adaptive_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = TemporalEncoder(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )
        self.graph = MultiScaleDynamicGraph(
            hidden_dim=hidden_dim,
            graph_heads=graph_heads,
            num_clusters=num_clusters,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [AdaptiveGatedConv(hidden_dim=hidden_dim, dilation=2**min(i + 1, 3), dropout=dropout)
             for i in range(num_adaptive_blocks)]
        )

    def forward(self, x, tod, dow, topology):
        h = self.temporal(x, tod, dow)
        aux = None
        for block in self.blocks:
            x_ms, aux = self.graph(h, topology)
            h = block(h, x_ms)
        return h, aux


class DynaFluxNet(nn.Module):
    """Multi-period, multi-scale dynamic graph traffic-flow forecaster."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        hidden_dim: int = 64,
        horizon: int = 12,
        temporal_heads: int = 4,
        temporal_layers: int = 3,
        graph_heads: int = 4,
        num_clusters: int = 41,
        num_adaptive_blocks: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.branches = nn.ModuleDict({
            name: DynaFluxBranch(
                num_nodes=num_nodes,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                temporal_heads=temporal_heads,
                temporal_layers=temporal_layers,
                graph_heads=graph_heads,
                num_clusters=num_clusters,
                num_adaptive_blocks=num_adaptive_blocks,
                dropout=dropout,
            )
            for name in ("hour", "day", "week")
        })
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, horizon),
        )

    def forward(self, batch: dict[str, torch.Tensor], topology: torch.Tensor, return_aux: bool = False):
        branch_features = []
        auxiliary = {}
        for name, branch in self.branches.items():
            h, aux = branch(
                batch[name],
                batch[f"{name}_tod"],
                batch[f"{name}_dow"],
                topology,
            )
            branch_features.append(h[:, -1])  # [B,N,D]
            auxiliary[name] = aux

        fused = torch.cat(branch_features, dim=-1)
        pred = self.regressor(fused).permute(0, 2, 1).unsqueeze(-1)  # [B,H,N,1]
        if return_aux:
            return pred, auxiliary
        return pred
