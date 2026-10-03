from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import torch

from dynaflux.data.dataset import MultiPeriodTrafficDataset, PeriodSpec, build_dataloaders
from dynaflux.data.io import load_pems_npz
from dynaflux.data.preprocessing import TrafficScaler, impute_missing
from dynaflux.engine.evaluator import evaluate_model
from dynaflux.models.model import DynaFluxNet
from dynaflux.utils.checkpoint import load_checkpoint
from dynaflux.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pems08.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--npz", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dc, gc, mc, tc = cfg["data"], cfg["graph"], cfg["model"], cfg["train"]
    raw = load_pems_npz(args.npz or dc["npz"], key=dc.get("key", "data"), flow_channel=dc.get("flow_channel", 0))
    raw, _ = impute_missing(raw, zero_as_missing=dc.get("zero_as_missing", True))

    s = np.load(args.scaler, allow_pickle=True)
    scaler = TrafficScaler(str(s["mode"])); scaler.center_ = s["center"]; scaler.scale_ = s["scale"]
    data = scaler.transform(raw)
    topology = torch.tensor(np.load(args.graph)["adjacency"], dtype=torch.float32)

    spec = PeriodSpec(dc.get("lookback",12), dc.get("day_offset",288), dc.get("week_offset",2016), dc.get("horizon",12), dc.get("steps_per_day",288))
    dataset = MultiPeriodTrafficDataset(data, spec=spec, start_weekday=dc.get("start_weekday", 0))
    _, _, test_loader = build_dataloaders(dataset, tc.get("batch_size",16), dc.get("split",[0.6,0.2,0.2]), tc.get("num_workers",0))
    model = DynaFluxNet(data.shape[1], data.shape[-1], mc.get("hidden_dim",64), dc.get("horizon",12), mc.get("temporal_heads",4), mc.get("temporal_layers",3), mc.get("graph_heads",4), gc.get("num_clusters",41), mc.get("num_adaptive_blocks",1), mc.get("dropout",0.3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); load_checkpoint(args.checkpoint, model, map_location=device)
    metrics, _, _ = evaluate_model(model, test_loader, topology, device, scaler=scaler)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
