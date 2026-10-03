from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np

from dynaflux.data.graph import build_spatial_graph
from dynaflux.data.io import load_edge_csv, load_pems_npz
from dynaflux.data.preprocessing import TrafficScaler, impute_missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--scaler", choices=["robust", "standard"], default="robust")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = load_pems_npz(args.npz)
    raw, mask = impute_missing(raw, zero_as_missing=True)
    cut = int(raw.shape[0] * 0.6)
    scaler = TrafficScaler(args.scaler).fit(raw[:cut])
    scaled = scaler.transform(raw)
    src, dst, cost = load_edge_csv(args.edges)
    graph = build_spatial_graph(raw.shape[1], src, dst, cost, args.sigma, args.threshold, args.resolution)

    np.savez_compressed(out / "traffic.npz", data=scaled, mask=mask)
    np.savez_compressed(out / "scaler.npz", center=scaler.center_, scale=scaler.scale_, mode=scaler.mode)
    np.savez_compressed(out / "graph.npz", adjacency=graph.adjacency, community_ids=graph.community_ids, community_mask=graph.community_mask)
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
