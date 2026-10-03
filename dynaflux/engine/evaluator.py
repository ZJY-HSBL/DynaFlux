from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from dynaflux.utils.metrics import horizon_metrics


@torch.no_grad()
def evaluate_model(model, loader, topology, device, scaler=None):
    model.eval()
    topology = topology.to(device)
    predictions, targets = [], []
    for batch in tqdm(loader, leave=False, desc="test"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model(batch, topology)
        predictions.append(pred.cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
    return horizon_metrics(y_true, y_pred), y_true, y_pred
