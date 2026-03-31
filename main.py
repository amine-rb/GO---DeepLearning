"""Main entry point for the Go deep-learning comparative study.

Usage examples
--------------
# Quick architecture validation (synthetic data, 2 epochs):
    python main.py --phase 1 --epochs 2

# Full Phase 1 on real SGF data:
    python main.py --phase 1 --data-dir /path/to/sgf_files --epochs 10

# Phase 2 deep training:
    python main.py --phase 2 --data-dir /path/to/sgf_files --epochs 30

# Phase 3 modern architectures:
    python main.py --phase 3 --data-dir /path/to/sgf_files --epochs 30

# Parameter budget report only (no training):
    python main.py --report-params
"""

import argparse

import torch


def report_parameters():
    """Print the parameter count for every architecture."""
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
    from models.phase3.mobilenet_v2 import MobileNetV2
    from models.phase3.convnext import ConvNeXt
    from models.phase3.efficientformer import EfficientFormer
    from models.phase3.hybrid_net import HybridNet

    all_models = {
        # Phase 1
        "Phase 1 – MLP (baseline)": MLP,
        "Phase 1 – SimpleCNN": SimpleCNN,
        "Phase 1 – ResNet": ResNet,
        "Phase 1 – DenseNet": DenseNet,
        "Phase 1 – InceptionNet": InceptionNet,
        "Phase 1 – SENet": SENet,
        "Phase 1 – MobileNet": MobileNet,
        "Phase 1 – DilatedCNN": DilatedCNN,
        "Phase 1 – AttentionNet": AttentionNet,
        "Phase 1 – WideResNet": WideResNet,
        # Phase 3
        "Phase 3 – MobileNetV2": MobileNetV2,
        "Phase 3 – ConvNeXt": ConvNeXt,
        "Phase 3 – EfficientFormer": EfficientFormer,
        "Phase 3 – HybridNet (best)": HybridNet,
    }

    print("\n" + "="*55)
    print(f"{'Architecture':<36} {'Parameters':>10} {'OK?':>6}")
    print("-"*55)
    budget = 100_000
    for name, ModelClass in all_models.items():
        m = ModelClass()
        n = m.count_parameters()
        ok = "✓" if n <= budget else "✗"
        print(f"{name:<36} {n:>10,} {ok:>6}")
    print("="*55)
    print(f"Budget: {budget:,} parameters\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative study of deep neural networks for Go 19×19"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3],
        help="Experiment phase to run (1, 2, or 3)"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory of SGF training files")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--report-params", action="store_true",
                        help="Print parameter counts and exit")
    args = parser.parse_args()

    if args.report_params:
        report_parameters()
        return

    if args.phase is None:
        parser.print_help()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.phase == 1:
        from experiments.phase1 import run_phase1
        run_phase1(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_games=args.max_games,
            save_dir=f"{args.save_dir}/phase1",
        )
    elif args.phase == 2:
        from experiments.phase2 import run_phase2
        run_phase2(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_games=args.max_games,
            save_dir=f"{args.save_dir}/phase2",
        )
    elif args.phase == 3:
        from experiments.phase3 import run_phase3
        run_phase3(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_games=args.max_games,
            save_dir=f"{args.save_dir}/phase3",
        )


if __name__ == "__main__":
    main()
