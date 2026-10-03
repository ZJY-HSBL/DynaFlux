import numpy as np

from dynaflux.data.graph import build_spatial_graph


def test_spatial_graph_shapes():
    src = np.array([0, 1, 2, 3])
    dst = np.array([1, 2, 3, 0])
    cost = np.array([10, 10, 10, 10], dtype=np.float32)
    g = build_spatial_graph(4, src, dst, cost, sigma=1.0, similarity_threshold=0.0)
    assert g.adjacency.shape == (4, 4)
    assert g.community_ids.shape == (4,)
    assert np.all(np.diag(g.adjacency) == 1)
