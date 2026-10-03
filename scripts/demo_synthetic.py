from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from dynaflux.data.dataset import MultiPeriodTrafficDataset, PeriodSpec
from dynaflux.models.model import DynaFluxNet


def main():
    rng = np.random.default_rng(42)
    t, n = 2300, 12
    time = np.arange(t, dtype=np.float32)
    base = 150 + 60 * np.sin(2 * np.pi * time / 288) + 30 * np.sin(2 * np.pi * time / 2016)
    data = np.stack([base + i * 2 + rng.normal(0, 5, t) for i in range(n)], axis=1)[..., None].astype(np.float32)
    dataset = MultiPeriodTrafficDataset(data, PeriodSpec(), start_weekday=0)
    sample = dataset[0]
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    topology = torch.eye(n)
    topology += torch.roll(torch.eye(n), 1, dims=1) + torch.roll(torch.eye(n), -1, dims=1)
    topology = (topology > 0).float()

    model = DynaFluxNet(num_nodes=n, hidden_dim=32, temporal_heads=4, graph_heads=4, num_clusters=4, num_adaptive_blocks=1)
    model.eval()
    with torch.no_grad():
        pred, aux = model(batch, topology, return_aux=True)
    print("prediction shape:", tuple(pred.shape))
    print("hour multi-scale graph shape:", tuple(aux["hour"]["multi_scale_graph"].shape))

if __name__ == "__main__":
    main()
