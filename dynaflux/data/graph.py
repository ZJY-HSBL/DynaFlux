from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class SpatialGraph:
    adjacency: np.ndarray
    community_ids: np.ndarray
    community_mask: np.ndarray


def build_spatial_graph(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    cost: np.ndarray,
    sigma: float = 1.0,
    similarity_threshold: float = 0.5,
    resolution: float = 0.5,
    seed: int = 42,
) -> SpatialGraph:
    """Build a distance-aware topology graph and Louvain community mask.

    Distances are converted to Gaussian affinities. Louvain runs on the original
    distance graph using inverse-distance edge strength, then the topology mask
    is augmented with within-community connections.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    cost = np.asarray(cost, dtype=np.float32)

    valid_cost = cost[np.isfinite(cost) & (cost > 0)]
    scale = float(np.std(valid_cost)) if valid_cost.size else 1.0
    if scale < 1e-6:
        scale = 1.0
    sigma = max(float(sigma), 1e-6)
    affinity = np.exp(-((cost / scale) ** 2) / (sigma**2)).astype(np.float32)

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for s, d, w in zip(src, dst, affinity):
        if 0 <= s < num_nodes and 0 <= d < num_nodes and w >= similarity_threshold:
            adjacency[s, d] = max(adjacency[s, d], float(w))
            adjacency[d, s] = max(adjacency[d, s], float(w))
    np.fill_diagonal(adjacency, 1.0)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for s, d, c in zip(src, dst, cost):
        if 0 <= s < num_nodes and 0 <= d < num_nodes and np.isfinite(c):
            g.add_edge(int(s), int(d), weight=float(1.0 / (1.0 + max(c, 0.0))))

    communities = nx.community.louvain_communities(
        g, weight="weight", resolution=float(resolution), seed=seed
    )
    community_ids = np.full(num_nodes, -1, dtype=np.int64)
    for cid, nodes in enumerate(communities):
        for node in nodes:
            community_ids[int(node)] = cid
    # Isolated nodes are assigned singleton communities.
    next_id = len(communities)
    for node in np.where(community_ids < 0)[0]:
        community_ids[node] = next_id
        next_id += 1

    community_mask = (community_ids[:, None] == community_ids[None, :]).astype(np.float32)
    topology = np.maximum(adjacency, community_mask)
    np.fill_diagonal(topology, 1.0)
    return SpatialGraph(topology.astype(np.float32), community_ids, community_mask)
