from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def impute_missing(x: np.ndarray, zero_as_missing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Create a validity mask and fill missing/zero values with per-node/channel means."""
    x = np.asarray(x, dtype=np.float32).copy()
    invalid = ~np.isfinite(x)
    if zero_as_missing:
        invalid |= x == 0
    mask = (~invalid).astype(np.float32)

    safe = np.where(invalid, np.nan, x)
    means = np.nanmean(safe, axis=0, keepdims=True)
    means = np.where(np.isfinite(means), means, 0.0)
    x = np.where(invalid, means, x)
    return x.astype(np.float32), mask


@dataclass
class TrafficScaler:
    mode: str = "robust"
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TrafficScaler":
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "robust":
            q25 = np.percentile(x, 25, axis=0, keepdims=True)
            q75 = np.percentile(x, 75, axis=0, keepdims=True)
            self.center_ = np.median(x, axis=0, keepdims=True)
            self.scale_ = q75 - q25
        elif self.mode in {"standard", "zscore"}:
            self.center_ = np.mean(x, axis=0, keepdims=True)
            self.scale_ = np.std(x, axis=0, keepdims=True)
        else:
            raise ValueError("mode must be 'robust' or 'standard'")
        self.scale_ = np.where(np.abs(self.scale_) < 1e-6, 1.0, self.scale_).astype(np.float32)
        self.center_ = self.center_.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        # Stored statistics have shape [1,N,C]. Broadcasting works for [B,H,N,C].
        return (x * self.scale_ + self.center_).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mode": self.mode, "center": self.center_, "scale": self.scale_}

    @classmethod
    def from_state_dict(cls, state: dict) -> "TrafficScaler":
        obj = cls(mode=state["mode"])
        obj.center_ = np.asarray(state["center"], dtype=np.float32)
        obj.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return obj
