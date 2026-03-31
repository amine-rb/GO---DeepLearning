"""Tests for the SGF parser and dataset classes."""

import pytest
import torch

from data.sgf_parser import parse_sgf, sgf_coord_to_index, BOARD_SIZE
from data.dataset import GoDataset, generate_synthetic_dataset, NUM_PLANES


# Minimal valid SGF game (5 moves, black wins)
SAMPLE_SGF = """(;GM[1]FF[4]SZ[19]RE[B+1.5]
;B[pd];W[dp];B[pp];W[dd];B[fc])"""


def test_sgf_coord_to_index():
    assert sgf_coord_to_index("aa") == 0
    assert sgf_coord_to_index("ba") == 1
    assert sgf_coord_to_index("ab") == BOARD_SIZE
    assert sgf_coord_to_index("ss") == 360   # 's'==18 → bottom-right corner
    assert sgf_coord_to_index("tt") is None  # 't' not in alphabet → pass


def test_parse_sgf_returns_examples():
    examples = parse_sgf(SAMPLE_SGF)
    assert len(examples) == 5, f"Expected 5 moves, got {len(examples)}"


def test_parse_sgf_example_structure():
    examples = parse_sgf(SAMPLE_SGF)
    features, policy, value = examples[0]
    assert len(features) == NUM_PLANES
    assert all(len(plane) == BOARD_SIZE * BOARD_SIZE for plane in features)
    assert isinstance(policy, int)
    assert 0 <= policy < BOARD_SIZE * BOARD_SIZE
    assert value in (-1.0, 0.0, 1.0)


def test_parse_sgf_value_from_current_perspective():
    examples = parse_sgf(SAMPLE_SGF)
    # Black moves first, game result is B+ → black's value should be +1
    _, _, v_black = examples[0]
    assert v_black == 1.0
    # White's first move → value should be -1 (from white's perspective vs B+ result)
    _, _, v_white = examples[1]
    assert v_white == -1.0


def test_go_dataset_from_examples():
    examples_raw = parse_sgf(SAMPLE_SGF)
    dataset = GoDataset(examples_raw)
    assert len(dataset) == 5
    x, p, v = dataset[0]
    assert x.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert x.dtype == torch.float32
    assert p.dtype == torch.long
    assert v.dtype == torch.float32


def test_synthetic_dataset():
    ds = generate_synthetic_dataset(num_samples=100)
    assert len(ds) == 100
    x, p, v = ds[0]
    assert x.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert 0 <= p.item() < BOARD_SIZE * BOARD_SIZE


def test_dataset_random_split():
    examples_raw = parse_sgf(SAMPLE_SGF * 10)  # replicate for more samples
    train_ds, val_ds = GoDataset.random_split(examples_raw, train_ratio=0.8)
    total = len(train_ds) + len(val_ds)
    assert total == len(examples_raw)
    assert len(train_ds) > len(val_ds)
