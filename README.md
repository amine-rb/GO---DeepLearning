# GO---DeepLearning

Comparative study of **14 deep neural network architectures** for move prediction in the game of Go (19×19).

## Overview

This repository implements a two-headed Go prediction framework constrained to a **maximum of 100,000 parameters**. Each model outputs:

- **Policy head** – probability distribution over all 361 legal moves
- **Value head** – win probability for the current player ∈ [−1, 1]

The study is organised in three phases:

| Phase | Description | Models |
|-------|-------------|--------|
| 1 | Exploration of 10 architectural families | MLP, SimpleCNN, ResNet, DenseNet, InceptionNet, SENet, MobileNet, DilatedCNN, AttentionNet, WideResNet |
| 2 | Deep training of the top-3 Phase 1 models | ResNet, SENet, DilatedCNN |
| 3 | Modern efficient architectures | MobileNetV2, ConvNeXt, EfficientFormer, HybridNet |

The best model (**HybridNet**, CNN + self-attention) achieves **44.56% policy accuracy** on the validation set, a gain of more than 43 percentage points over the MLP baseline.

---

## Project Structure

```
GO---DeepLearning/
├── data/
│   ├── dataset.py         # GoDataset (SGF files or synthetic)
│   └── sgf_parser.py      # SGF parser & board simulator
├── models/
│   ├── base_model.py      # GoModel base class
│   ├── phase1/            # 10 Phase 1 architectures
│   │   ├── mlp.py
│   │   ├── simple_cnn.py
│   │   ├── resnet.py
│   │   ├── densenet.py
│   │   ├── inception.py
│   │   ├── senet.py
│   │   ├── mobilenet.py
│   │   ├── dilated_cnn.py
│   │   ├── attention_net.py
│   │   └── wide_resnet.py
│   └── phase3/            # 4 Phase 3 architectures
│       ├── mobilenet_v2.py
│       ├── convnext.py
│       ├── efficientformer.py
│       └── hybrid_net.py  # Best model
├── training/
│   ├── trainer.py         # Training loop with cosine LR schedule
│   └── losses.py          # Dual-head combined loss
├── experiments/
│   ├── phase1.py          # Phase 1 experiment runner
│   ├── phase2.py          # Phase 2 experiment runner
│   └── phase3.py          # Phase 3 experiment runner
├── utils/
│   └── metrics.py         # Policy accuracy, top-k accuracy, value MAE
├── tests/                 # pytest test suite
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── main.py                # Unified CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Parameter Budget Report

```bash
python main.py --report-params
```

### Training

```bash
# Phase 1 – explore 10 architectures (synthetic data for quick test)
python main.py --phase 1 --epochs 10

# Phase 1 – with real SGF game data
python main.py --phase 1 --data-dir /path/to/sgf_files --epochs 10

# Phase 2 – deep training of top-3
python main.py --phase 2 --data-dir /path/to/sgf_files --epochs 30

# Phase 3 – modern efficient architectures
python main.py --phase 3 --data-dir /path/to/sgf_files --epochs 30
```

### Running Tests

```bash
python -m pytest tests/ -v
```

---

## Model Architecture Summary

| Architecture | Params | Key Idea |
|---|---|---|
| MLP | 97,730 | Fully-connected baseline |
| SimpleCNN | 80,898 | Plain 3×3 conv stack |
| ResNet | 96,226 | Residual connections |
| DenseNet | 88,612 | Dense skip connections |
| InceptionNet | 15,274 | Parallel multi-scale convolutions |
| SENet | 97,686 | Squeeze-and-Excitation attention |
| MobileNet | 22,642 | Depthwise separable convolutions |
| DilatedCNN | 74,134 | Multi-scale dilated convolutions |
| AttentionNet | 43,619 | Spatial self-attention |
| WideResNet | 72,962 | Wider but shallower residual network |
| MobileNetV2 | 79,454 | Inverted residuals (Phase 3) |
| ConvNeXt | 64,930 | Modernised CNN with LayerNorm (Phase 3) |
| EfficientFormer | 83,850 | MetaFormer with pooling mixer (Phase 3) |
| **HybridNet** | **72,610** | **CNN + self-attention – best model** (Phase 3) |

All models use the same dual-head output convention:
- Policy head: `Conv2d(channels, 1, 1)` → flatten to 361 logits
- Value head: `AdaptiveAvgPool2d(1)` → `Linear(channels, 64)` → `Linear(64, 1)` → `Tanh`

---

## Data Format

The framework accepts **SGF** (Smart Game Format) files as training data. Place all `.sgf` files in a directory and pass it with `--data-dir`. Each board position generates one training example.

For quick architecture validation without real game data, a synthetic dataset generator is built in (activated when `--data-dir` is omitted).

### Input Representation

Each board position is encoded as 4 binary planes (4 × 19 × 19):

| Plane | Content |
|-------|---------|
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Ko point (if any) |
| 3 | Constant 1 (board indicator) |

---

## Training Details

- **Optimiser**: Adam with weight decay 1e-4
- **LR Schedule**: Cosine annealing (T_max = 50)
- **Loss**: `policy_weight × CrossEntropy(policy) + value_weight × MSE(value)`
- **Gradient clipping**: max_norm = 1.0
- **Validation split**: 80 % train / 20 % validation

---

## Results

The best policy accuracy on the KGS dataset validation set was achieved by **HybridNet** at **44.56%**, compared to the MLP baseline at ~1.4%. The CNN + self-attention combination captures both local tactical patterns and global strategic context.
