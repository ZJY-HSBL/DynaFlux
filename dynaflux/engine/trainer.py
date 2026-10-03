from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from dynaflux.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        topology: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 5.0,
        milestones=(20, 30),
        gamma: float = 0.3,
        huber_delta: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.topology = topology.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(milestones), gamma=gamma
        )
        self.loss_fn = nn.HuberLoss(delta=huber_delta)
        self.grad_clip = grad_clip

    def _move_batch(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        total_items = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, leave=False, desc="train" if train else "valid"):
                batch = self._move_batch(batch)
                target = batch["target"]
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch, self.topology)
                loss = self.loss_fn(pred, target)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                bs = target.shape[0]
                total_loss += float(loss.detach()) * bs
                total_items += bs
        if train:
            self.scheduler.step()
        return total_loss / max(total_items, 1)

    def fit(self, train_loader, val_loader, epochs: int, output_dir: str | Path, patience: int = 10):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best = float("inf")
        bad_epochs = 0
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self.run_epoch(train_loader, train=True)
            val_loss = self.run_epoch(val_loader, train=False)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")
            save_checkpoint(output_dir / "last.pt", self.model, self.optimizer, self.scheduler, epoch)
            if val_loss < best:
                best = val_loss
                bad_epochs = 0
                save_checkpoint(output_dir / "best.pt", self.model, self.optimizer, self.scheduler, epoch)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        return history
