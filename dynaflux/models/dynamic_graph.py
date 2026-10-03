from __future__ import annotations

import torch
from torch import nn

from .gat import TopologyAttentionGraph


def _kmeans(x: torch.Tensor, k: int, iterations: int = 10) -> torch.Tensor:
    """Small deterministic k-means used for spectral node grouping."""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return torch.zeros(n, dtype=torch.long, device=x.device)
    # Deterministic spread initialization avoids host-side dependencies.
    init_idx = torch.linspace(0, n - 1, k, device=x.device).round().long()
    centers = x[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iterations):
        dist = torch.cdist(x, centers)
        new_labels = dist.argmin(dim=1)
        new_centers = []
        for cid in range(k):
            members = x[new_labels == cid]
            new_centers.append(members.mean(dim=0) if members.numel() else centers[cid])
        centers = torch.stack(new_centers, dim=0)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    return labels


class GlobalDynamicAssociation(nn.Module):
    """Flow-similarity spectral graph: distance -> similarity -> Laplacian -> clustering."""

    def __init__(self, num_clusters: int = 41, kmeans_iterations: int = 8) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iterations = kmeans_iterations

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,N,D]. Flatten time/features per node.
        b, t, n, d = x.shape
        series = x.permute(0, 2, 1, 3).reshape(b, n, t * d)
        distances = torch.cdist(series, series, p=2)
        similarity = 1.0 / (1.0 + distances)
        eye = torch.eye(n, device=x.device, dtype=x.dtype)[None, :, :]
        similarity = similarity * (1.0 - eye) + eye

        degree = similarity.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt[:, :, None] * similarity * inv_sqrt[:, None, :]
        laplacian = eye - normalized

        graphs = []
        all_labels = []
        k = max(1, min(self.num_clusters, n))
        for batch_idx in range(b):
            _, eigvecs = torch.linalg.eigh(laplacian[batch_idx].float())
            embed = eigvecs[:, :k]
            labels = _kmeans(embed, k=k, iterations=self.kmeans_iterations)
            co_cluster = (labels[:, None] == labels[None, :]).to(x.dtype)
            graphs.append(co_cluster)
            all_labels.append(labels)
        return torch.stack(graphs, dim=0), torch.stack(all_labels, dim=0)


class MultiScaleDynamicGraph(nn.Module):
    """Fuse global flow association and local topology association."""

    def __init__(
        self,
        hidden_dim: int,
        graph_heads: int = 4,
        num_clusters: int = 41,
        fusion_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_graph = GlobalDynamicAssociation(num_clusters=num_clusters)
        self.local_graph = TopologyAttentionGraph(hidden_dim, num_heads=graph_heads, dropout=dropout)
        self.fusion_temperature = max(float(fusion_temperature), 1e-4)

    def forward(self, x: torch.Tensor, topology: torch.Tensor) -> tuple[torch.Tensor, dict]:
        a_global, labels = self.global_graph(x)
        a_local = self.local_graph(x, topology)
        fused_logits = (a_global * a_local) / self.fusion_temperature
        a_multi = torch.softmax(fused_logits, dim=-1)
        x_ms = torch.einsum("bij,btjd->btid", a_multi, x)
        return x_ms, {
            "global_graph": a_global,
            "local_graph": a_local,
            "multi_scale_graph": a_multi,
            "cluster_labels": labels,
        }
