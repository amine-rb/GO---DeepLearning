"""PyTorch dataset classes for Go game prediction.

Supports loading from SGF files or from pre-generated synthetic data
for architecture validation and quick testing.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.sgf_parser import load_sgf_directory, load_sgf_file, BOARD_SIZE

# Number of input feature planes
NUM_PLANES = 4


def features_to_tensor(features: list) -> torch.Tensor:
    """Convert raw feature planes (list of flat lists) to a CHW tensor."""
    arr = np.array(features, dtype=np.float32)  # (4, 361)
    arr = arr.reshape(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    return torch.from_numpy(arr)


class GoDataset(Dataset):
    """Dataset of Go board positions for two-headed (policy + value) prediction.

    Each item is a tuple:
        (features, policy_target, value_target)
    where:
        features      - float32 tensor of shape (NUM_PLANES, 19, 19)
        policy_target - int64 tensor, scalar in [0, 360]
        value_target  - float32 tensor, scalar in [-1, 1]
    """

    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, policy, value = self.examples[idx]
        x = features_to_tensor(features)
        p = torch.tensor(policy, dtype=torch.long)
        v = torch.tensor(value, dtype=torch.float32)
        return x, p, v

    @classmethod
    def from_sgf_directory(cls, directory: str, max_games: Optional[int] = None) -> "GoDataset":
        examples = load_sgf_directory(directory, max_games=max_games)
        return cls(examples)

    @classmethod
    def from_sgf_file(cls, filepath: str) -> "GoDataset":
        examples = load_sgf_file(filepath)
        return cls(examples)

    @classmethod
    def random_split(cls, examples: list, train_ratio: float = 0.8, seed: int = 42):
        """Split examples into train and validation datasets."""
        rng = random.Random(seed)
        shuffled = examples[:]
        rng.shuffle(shuffled)
        split = int(len(shuffled) * train_ratio)
        return cls(shuffled[:split]), cls(shuffled[split:])


def generate_synthetic_dataset(num_samples: int = 10000, seed: int = 42) -> GoDataset:
    """Generate a synthetic dataset of random Go positions for testing.

    This is used for architecture validation and unit tests when no SGF data
    is available.  Labels are random, so accuracy is expected to be close to
    chance (≈1/361 ≈ 0.28 % for policy).
    """
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(num_samples):
        # Random board with ~30 % stones
        board = rng.random((NUM_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
        board[0] = (board[0] > 0.7).astype(np.float32)   # current stones
        board[1] = (board[1] > 0.7).astype(np.float32)   # opponent stones
        board[2] = (board[2] > 0.99).astype(np.float32)  # ko (rare)
        board[3] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        policy = int(rng.integers(0, BOARD_SIZE * BOARD_SIZE))
        value = float(rng.choice([-1.0, 0.0, 1.0]))
        examples.append((board, policy, value))

    # Wrap in dataset
    class _SyntheticDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, idx):
            board, policy, value = examples[idx]
            if isinstance(board, np.ndarray):
                x = torch.from_numpy(board)
            else:
                x = board
            return x, torch.tensor(policy, dtype=torch.long), torch.tensor(value, dtype=torch.float32)

    return _SyntheticDataset()
