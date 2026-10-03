from __future__ import annotations

import numpy as np


def _flat(y_true, y_pred):
    a = np.asarray(y_true, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(y_true, y_pred) -> float:
    a, b = _flat(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _flat(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-5) -> float:
    a, b = _flat(y_true, y_pred)
    mask = np.abs(a) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)


def r2_score(y_true, y_pred) -> float:
    a, b = _flat(y_true, y_pred)
    denom = np.sum((a - a.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def horizon_metrics(y_true, y_pred, horizons=(3, 6, 9, 12)) -> dict:
    result = {}
    max_h = min(y_true.shape[1], y_pred.shape[1])
    for h in horizons:
        if h <= max_h:
            yt = y_true[:, h - 1 : h]
            yp = y_pred[:, h - 1 : h]
            result[f"t{h}"] = {
                "MAE": mae(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAPE": mape(yt, yp),
                "R2": r2_score(yt, yp),
            }
    result["average"] = {
        "MAE": mae(y_true[:, :max_h], y_pred[:, :max_h]),
        "RMSE": rmse(y_true[:, :max_h], y_pred[:, :max_h]),
        "MAPE": mape(y_true[:, :max_h], y_pred[:, :max_h]),
        "R2": r2_score(y_true[:, :max_h], y_pred[:, :max_h]),
    }
    return result
