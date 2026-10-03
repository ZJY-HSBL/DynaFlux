from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import torch

from dynaflux.models.model import DynaFluxNet
from dynaflux.utils.checkpoint import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Run DynaFlux inference from a prepared NPZ batch.")
    p.add_argument("--batch", required=True, help="NPZ with hour/day/week, *_tod, *_dow arrays")
    p.add_argument("--graph", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-clusters", type=int, default=41)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num-adaptive-blocks", type=int, default=1)
    args = p.parse_args()

    z = np.load(args.batch)
    keys = ["hour","day","week","hour_tod","hour_dow","day_tod","day_dow","week_tod","week_dow"]
    batch = {k: torch.tensor(z[k], dtype=torch.float32) for k in keys}
    topology = torch.tensor(np.load(args.graph)["adjacency"], dtype=torch.float32)
    model = DynaFluxNet(args.num_nodes, hidden_dim=args.hidden_dim, horizon=args.horizon, num_clusters=args.num_clusters, num_adaptive_blocks=args.num_adaptive_blocks)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    with torch.no_grad():
        pred = model(batch, topology).numpy()
    print(json.dumps({"shape": list(pred.shape), "prediction": pred.tolist()}))

if __name__ == "__main__":
    main()
