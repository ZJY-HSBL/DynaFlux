from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dynaflux.data.dataset import MultiPeriodTrafficDataset, PeriodSpec, build_dataloaders
from dynaflux.data.graph import build_spatial_graph
from dynaflux.data.io import load_edge_csv, load_pems_npz
from dynaflux.data.preprocessing import TrafficScaler, impute_missing
from dynaflux.engine.evaluator import evaluate_model
from dynaflux.engine.trainer import Trainer
from dynaflux.models.model import DynaFluxNet
from dynaflux.utils.checkpoint import load_checkpoint
from dynaflux.utils.config import load_config
from dynaflux.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pems08.yaml")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--edges", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    dc, gc, mc, tc = cfg["data"], cfg["graph"], cfg["model"], cfg["train"]
    npz_path = args.npz or dc["npz"]
    edge_path = args.edges or dc["edges"]

    raw = load_pems_npz(npz_path, key=dc.get("key", "data"), flow_channel=dc.get("flow_channel", 0))
    raw, _ = impute_missing(raw, zero_as_missing=dc.get("zero_as_missing", True))

    # Fit statistics on the first 60% of the timeline to avoid test leakage.
    train_cut = max(1, int(raw.shape[0] * float(dc.get("split", [0.6, 0.2, 0.2])[0])))
    scaler = TrafficScaler(dc.get("scaler", "robust")).fit(raw[:train_cut])
    data = scaler.transform(raw)

    src, dst, cost = load_edge_csv(edge_path)
    graph = build_spatial_graph(
        num_nodes=data.shape[1],
        src=src,
        dst=dst,
        cost=cost,
        sigma=gc.get("sigma", 1.0),
        similarity_threshold=gc.get("similarity_threshold", 0.5),
        resolution=gc.get("resolution", 0.5),
        seed=cfg.get("seed", 42),
    )
    topology = torch.tensor(graph.adjacency, dtype=torch.float32)

    spec = PeriodSpec(
        lookback=dc.get("lookback", 12),
        horizon=dc.get("horizon", 12),
        day_offset=dc.get("day_offset", 288),
        week_offset=dc.get("week_offset", 2016),
        steps_per_day=dc.get("steps_per_day", 288),
    )
    dataset = MultiPeriodTrafficDataset(data, spec=spec, start_weekday=dc.get("start_weekday", 0))
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        batch_size=tc.get("batch_size", 16),
        split=dc.get("split", [0.6, 0.2, 0.2]),
        num_workers=tc.get("num_workers", 0),
    )

    model = DynaFluxNet(
        num_nodes=data.shape[1],
        in_dim=data.shape[-1],
        hidden_dim=mc.get("hidden_dim", 64),
        horizon=dc.get("horizon", 12),
        temporal_heads=mc.get("temporal_heads", 4),
        temporal_layers=mc.get("temporal_layers", 3),
        graph_heads=mc.get("graph_heads", 4),
        num_clusters=gc.get("num_clusters", 41),
        num_adaptive_blocks=mc.get("num_adaptive_blocks", 1),
        dropout=mc.get("dropout", 0.3),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        topology=topology,
        device=device,
        lr=tc.get("learning_rate", 1e-3),
        weight_decay=tc.get("weight_decay", 0.0),
        grad_clip=tc.get("grad_clip", 5.0),
        milestones=tc.get("milestones", [20, 30]),
        gamma=tc.get("lr_gamma", 0.3),
        huber_delta=tc.get("huber_delta", 1.0),
    )
    out_dir = Path(tc.get("output_dir", "outputs/run"))
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=tc.get("epochs", 40),
        output_dir=out_dir,
        patience=tc.get("patience", 10),
    )

    load_checkpoint(out_dir / "best.pt", model, map_location=device)
    metrics, _, _ = evaluate_model(model, test_loader, topology, device, scaler=scaler)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.savez(out_dir / "scaler.npz", center=scaler.center_, scale=scaler.scale_, mode=scaler.mode)
    np.savez(
        out_dir / "graph.npz",
        adjacency=graph.adjacency,
        community_ids=graph.community_ids,
        community_mask=graph.community_mask,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
