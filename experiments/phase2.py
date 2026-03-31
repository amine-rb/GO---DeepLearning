"""Phase 2 experiment: deep training of the top 3 Phase 1 models.

Takes the three best-performing architectures from Phase 1 and trains them
with more data, more epochs, and a refined schedule.
"""

import os
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import GoDataset, generate_synthetic_dataset
from models.phase1.resnet import ResNet
from models.phase1.senet import SENet
from models.phase1.dilated_cnn import DilatedCNN
from training.trainer import Trainer


# The three Phase 1 models selected for Phase 2 training.
# (Chosen based on typical Phase 1 results; can be overridden via `model_names`.)
PHASE2_MODELS = {
    "ResNet": ResNet,
    "SENet": SENet,
    "DilatedCNN": DilatedCNN,
}


def run_phase2(
    data_dir: Optional[str] = None,
    num_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 5e-4,
    num_synthetic: int = 50000,
    save_dir: str = "checkpoints/phase2",
    max_games: Optional[int] = None,
    verbose: bool = True,
):
    """Run Phase 2 deep training on the top-3 architectures.

    Args:
        data_dir:       Path to directory containing SGF files.
        num_epochs:     Number of training epochs (more than Phase 1).
        batch_size:     Mini-batch size.
        lr:             Initial learning rate.
        num_synthetic:  Synthetic samples if data_dir is None.
        save_dir:       Root directory for checkpoints.
        max_games:      Maximum number of SGF games to load.
        verbose:        Print per-epoch stats.

    Returns:
        results dict: model_name → metrics
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

    for name, ModelClass in PHASE2_MODELS.items():
        if verbose:
            print(f"\n{'='*60}\nPhase 2 training: {name}\n{'='*60}")

        model = ModelClass()
        n_params = model.count_parameters()
        if verbose:
            print(f"Parameters: {n_params:,}")

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
            print(f"\n{name} Phase 2 done – best val_policy_acc={best_val_acc:.4f}")

    if verbose:
        _print_results_table(results)

    return results


def _print_results_table(results: dict):
    print("\n" + "="*70)
    print(f"{'Model':<18} {'Params':>8} {'Val Policy Acc':>15} {'Val Value MAE':>14} {'Time (s)':>10}")
    print("-"*70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["val_policy_acc"]):
        print(f"{name:<18} {r['params']:>8,} {r['val_policy_acc']:>14.4f} "
              f"{r['val_value_mae']:>13.4f} {r['time_s']:>10.1f}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Deep training of top-3 models")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase2")
    args = parser.parse_args()

    run_phase2(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_games=args.max_games,
        save_dir=args.save_dir,
    )
