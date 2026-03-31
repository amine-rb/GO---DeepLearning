"""Training loop for dual-head Go prediction models."""

import os
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import DualHeadLoss
from utils.metrics import compute_metrics


class Trainer:
    """Encapsulates the training loop for any GoModel.

    Args:
        model:          GoModel instance
        train_loader:   DataLoader for training data
        val_loader:     DataLoader for validation data
        lr:             Initial learning rate
        policy_weight:  Weight for the policy loss term
        value_weight:   Weight for the value loss term
        device:         'cuda' or 'cpu'
        save_dir:       Directory for saving checkpoints (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        device: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = DualHeadLoss(policy_weight, value_weight)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_policy_acc": [], "val_policy_acc": [],
            "train_value_mae": [], "val_value_mae": [],
        }

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        total_policy_acc = 0.0
        total_value_mae = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, policy_target, value_target in loader:
                x = x.to(self.device)
                policy_target = policy_target.to(self.device)
                value_target = value_target.to(self.device)

                policy_logits, value_pred = self.model(x)
                loss, _, _ = self.criterion(
                    policy_logits, value_pred, policy_target, value_target
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                metrics = compute_metrics(policy_logits, value_pred,
                                          policy_target, value_target)
                total_loss += loss.item()
                total_policy_acc += metrics["policy_acc"]
                total_value_mae += metrics["value_mae"]
                n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "policy_acc": total_policy_acc / max(n_batches, 1),
            "value_mae": total_value_mae / max(n_batches, 1),
        }

    def train(self, num_epochs: int, verbose: bool = True) -> Dict[str, list]:
        """Train for `num_epochs` epochs and return training history."""
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_stats = self._run_epoch(self.train_loader, train=True)
            val_stats = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["train_policy_acc"].append(train_stats["policy_acc"])
            self.history["val_policy_acc"].append(val_stats["policy_acc"])
            self.history["train_value_mae"].append(train_stats["value_mae"])
            self.history["val_value_mae"].append(val_stats["value_mae"])

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_stats['loss']:.4f} "
                    f"val_loss={val_stats['loss']:.4f} | "
                    f"train_pol_acc={train_stats['policy_acc']:.4f} "
                    f"val_pol_acc={val_stats['policy_acc']:.4f} | "
                    f"val_val_mae={val_stats['value_mae']:.4f} | "
                    f"time={elapsed:.1f}s"
                )

            # Save best model
            if self.save_dir and val_stats["policy_acc"] > best_val_acc:
                best_val_acc = val_stats["policy_acc"]
                path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), path)

        return self.history

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the validation set."""
        return self._run_epoch(self.val_loader, train=False)
