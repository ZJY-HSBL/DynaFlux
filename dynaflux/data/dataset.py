from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class PeriodSpec:
    lookback: int = 12
    day_offset: int = 288
    week_offset: int = 2016
    horizon: int = 12
    steps_per_day: int = 288


class MultiPeriodTrafficDataset(Dataset):
    """Hour/day/week aligned samples for 5-minute traffic-flow series."""

    def __init__(self, data: np.ndarray, spec: PeriodSpec | None = None, start_weekday: int = 4):
        if data.ndim == 2:
            data = data[..., None]
        if data.ndim != 3:
            raise ValueError(f"Expected [T,N,C], got {data.shape}")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.spec = spec or PeriodSpec()
        self.start_weekday = int(start_weekday) % 7
        s = self.spec
        start = max(s.lookback, s.day_offset + s.lookback, s.week_offset + s.lookback)
        stop = data.shape[0] - s.horizon + 1
        self.targets = np.arange(start, stop, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def _calendar(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        tod = (indices % self.spec.steps_per_day) / float(self.spec.steps_per_day - 1)
        dow = ((indices // self.spec.steps_per_day + self.start_weekday) % 7) / 6.0
        return torch.tensor(tod, dtype=torch.float32), torch.tensor(dow, dtype=torch.float32)

    def _window(self, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.arange(end - self.spec.lookback, end, dtype=np.int64)
        tod, dow = self._calendar(idx)
        return self.data[idx], tod, dow

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t = int(self.targets[index])
        h, h_tod, h_dow = self._window(t)
        d, d_tod, d_dow = self._window(t - self.spec.day_offset)
        w, w_tod, w_dow = self._window(t - self.spec.week_offset)
        y = self.data[t : t + self.spec.horizon]
        return {
            "hour": h,
            "day": d,
            "week": w,
            "hour_tod": h_tod,
            "hour_dow": h_dow,
            "day_tod": d_tod,
            "day_dow": d_dow,
            "week_tod": w_tod,
            "week_dow": w_dow,
            "target": y,
        }


def _split_indices(n: int, ratios: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(list(ratios), dtype=np.float64)
    r = r / r.sum()
    n_train = int(n * r[0])
    n_val = int(n * r[1])
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def build_dataloaders(
    dataset: Dataset,
    batch_size: int = 16,
    split=(0.6, 0.2, 0.2),
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = _split_indices(len(dataset), split)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(Subset(dataset, train_idx), shuffle=True, **common),
        DataLoader(Subset(dataset, val_idx), shuffle=False, **common),
        DataLoader(Subset(dataset, test_idx), shuffle=False, **common),
    )
