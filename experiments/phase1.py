"""Phase 1 experiment: exploration of 10 architectural families.

Trains all 10 Phase-1 models for a configurable number of epochs and
reports comparative metrics (policy accuracy, parameter count, training time).
"""

import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import GoDataset, generate_synthetic_dataset
from models.phase1.mlp import MLP
from models.phase1.simple_cnn import SimpleCNN
from models.phase1.resnet import ResNet
from models.phase1.densenet import DenseNet
from models.phase1.inception import InceptionNet
from models.phase1.senet import SENet
from models.phase1.mobilenet import MobileNet
from models.phase1.dilated_cnn import DilatedCNN
from models.phase1.attention_net import AttentionNet
from models.phase1.wide_resnet import WideResNet
from training.trainer import Trainer


PHASE1_MODELS = {
    "MLP": MLP,
    "SimpleCNN": SimpleCNN,
    "ResNet": ResNet,
    "DenseNet": DenseNet,
    "InceptionNet": InceptionNet,
    "SENet": SENet,
    "MobileNet": MobileNet,
    "DilatedCNN": DilatedCNN,
    "AttentionNet": AttentionNet,
    "WideResNet": WideResNet,
}


def run_phase1(
    data_dir: Optional[str] = None,
    num_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_synthetic: int = 20000,
    save_dir: str = "checkpoints/phase1",
    max_games: Optional[int] = None,
    verbose: bool = True,
):
    """Run Phase 1 experiment.

    Args:
        data_dir:       Path to directory containing SGF files. If None, uses
                        synthetic data for a quick architecture validation run.
        num_epochs:     Number of training epochs per model.
        batch_size:     Mini-batch size.
        lr:             Initial learning rate.
        num_synthetic:  Number of synthetic samples to use if data_dir is None.
        save_dir:       Root directory for model checkpoints.
        max_games:      Maximum number of SGF games to load (None = all).
        verbose:        Print per-epoch stats.

    Returns:
        results dict: model_name → {params, val_policy_acc, val_value_mae, time_s}
    """
    # ── Data ────────────────────────────────────────────────────────────────
    if data_dir is not None:
        dataset = GoDataset.from_sgf_directory(data_dir, max_games=max_games)
    else:
        if verbose:
            print(f"No data_dir provided – using {num_synthetic} synthetic samples.")
        dataset = generate_synthetic_dataset(num_synthetic)

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    results = {}

    for name, ModelClass in PHASE1_MODELS.items():
        if verbose:
            print(f"\n{'='*60}\nTraining {name}\n{'='*60}")

        model = ModelClass()
        n_params = model.count_parameters()
        if verbose:
            print(f"Parameters: {n_params:,}")
            if not model.parameter_budget_ok():
                print(f"  ⚠ WARNING: {name} exceeds 100 k parameter budget!")

        model_save_dir = os.path.join(save_dir, name)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            save_dir=model_save_dir,
        )

        t0 = time.time()
        history = trainer.train(num_epochs=num_epochs, verbose=verbose)
        elapsed = time.time() - t0

        best_val_acc = max(history["val_policy_acc"])
        final_val_mae = history["val_value_mae"][-1]

        results[name] = {
            "params": n_params,
            "val_policy_acc": best_val_acc,
            "val_value_mae": final_val_mae,
            "time_s": elapsed,
        }

        if verbose:
            print(f"\n{name} done – best val_policy_acc={best_val_acc:.4f}, "
                  f"params={n_params:,}, time={elapsed:.1f}s")

    if verbose:
        _print_results_table(results)

    return results


def _print_results_table(results: dict):
    """Print a formatted comparison table."""
    print("\n" + "="*70)
    print(f"{'Model':<18} {'Params':>8} {'Val Policy Acc':>15} {'Val Value MAE':>14} {'Time (s)':>10}")
    print("-"*70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["val_policy_acc"]):
        print(f"{name:<18} {r['params']:>8,} {r['val_policy_acc']:>14.4f} "
              f"{r['val_value_mae']:>13.4f} {r['time_s']:>10.1f}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Explore 10 architectures")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase1")
    args = parser.parse_args()

    run_phase1(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_games=args.max_games,
        save_dir=args.save_dir,
    )
