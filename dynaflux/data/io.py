from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_pems_npz(path: str | Path, key: str = "data", flow_channel: int = 0) -> np.ndarray:
    """Load a PEMS-style NPZ tensor and return [time, nodes, 1] flow data."""
    path = Path(path)
    with np.load(path) as npz:
        if key not in npz:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(npz.keys())}")
        data = np.asarray(npz[key], dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"Expected [T,N,C], got {data.shape}")
    if not (0 <= flow_channel < data.shape[-1]):
        raise ValueError(f"flow_channel={flow_channel} out of range for {data.shape}")
    return data[..., flow_channel : flow_channel + 1]


def load_edge_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load edge triples (from, to, cost/distance) from a CSV file."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    src_col = cols.get("from") or cols.get("src") or cols.get("source")
    dst_col = cols.get("to") or cols.get("dst") or cols.get("target")
    cost_col = cols.get("cost") or cols.get("distance") or cols.get("dist")
    if not all([src_col, dst_col, cost_col]):
        if df.shape[1] < 3:
            raise ValueError("Edge CSV must contain at least three columns: from,to,cost")
        src_col, dst_col, cost_col = df.columns[:3]
    return (
        df[src_col].to_numpy(dtype=np.int64),
        df[dst_col].to_numpy(dtype=np.int64),
        df[cost_col].to_numpy(dtype=np.float32),
    )
