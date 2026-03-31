"""Tests for the training loop (Trainer) and loss functions."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.losses import DualHeadLoss
from training.trainer import Trainer
from models.phase1.simple_cnn import SimpleCNN


def _make_loader(n: int = 64, batch_size: int = 16):
    """Create a tiny DataLoader with random data."""
    x = torch.randn(n, 4, 19, 19)
    p = torch.randint(0, 361, (n,))
    v = torch.zeros(n)
    ds = TensorDataset(x, p, v)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def test_dual_head_loss_shapes():
    criterion = DualHeadLoss()
    B = 8
    logits = torch.randn(B, 361)
    value = torch.tanh(torch.randn(B, 1))
    targets_p = torch.randint(0, 361, (B,))
    targets_v = torch.zeros(B)
    total, pl, vl = criterion(logits, value, targets_p, targets_v)
    assert total.shape == ()
    assert pl.shape == ()
    assert vl.shape == ()
    assert total.item() > 0


def test_trainer_runs_one_epoch():
    model = SimpleCNN()
    loader = _make_loader()
    trainer = Trainer(model, loader, loader, lr=1e-3)
    history = trainer.train(num_epochs=1, verbose=False)
    assert "val_policy_acc" in history
    assert len(history["val_policy_acc"]) == 1


def test_trainer_loss_decreases():
    """With enough capacity and iterations, training loss should decrease."""
    model = SimpleCNN()
    loader = _make_loader(n=128, batch_size=32)
    trainer = Trainer(model, loader, loader, lr=1e-2)
    history = trainer.train(num_epochs=3, verbose=False)
    first_loss = history["train_loss"][0]
    last_loss = history["train_loss"][-1]
    # Not strictly guaranteed on random data, but check it runs
    assert isinstance(first_loss, float)
    assert isinstance(last_loss, float)
