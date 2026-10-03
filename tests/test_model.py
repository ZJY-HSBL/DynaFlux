import torch

from dynaflux.models.model import DynaFluxNet


def test_model_forward():
    b, t, n = 2, 12, 8
    batch = {}
    for name in ("hour", "day", "week"):
        batch[name] = torch.randn(b, t, n, 1)
        batch[f"{name}_tod"] = torch.rand(b, t)
        batch[f"{name}_dow"] = torch.rand(b, t)
    topology = torch.eye(n)
    topology += torch.roll(torch.eye(n), 1, dims=1) + torch.roll(torch.eye(n), -1, dims=1)
    topology = (topology > 0).float()
    model = DynaFluxNet(
        num_nodes=n,
        hidden_dim=16,
        temporal_heads=4,
        temporal_layers=1,
        graph_heads=4,
        num_clusters=3,
        num_adaptive_blocks=1,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        y = model(batch, topology)
    assert y.shape == (b, 12, n, 1)
    assert torch.isfinite(y).all()
