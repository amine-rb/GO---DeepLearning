"""Unit tests for all 14 neural network architectures.

Checks:
  1. All models run a forward pass without error.
  2. Output shapes are correct: (B, 361) policy + (B, 1) value.
  3. All models respect the 100 k parameter budget.
"""

import pytest
import torch

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

BATCH_SIZE = 4
NUM_PLANES = 4
BOARD_SIZE = 19
NUM_MOVES = 361
BUDGET = 100_000

ALL_MODELS = [
    MLP,
    SimpleCNN,
    ResNet,
    DenseNet,
    InceptionNet,
    SENet,
    MobileNet,
    DilatedCNN,
    AttentionNet,
    WideResNet,
    MobileNetV2,
    ConvNeXt,
    EfficientFormer,
    HybridNet,
]


@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(BATCH_SIZE, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)


@pytest.mark.parametrize("ModelClass", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_forward_pass(ModelClass, dummy_input):
    """Model forward pass should run without error."""
    model = ModelClass()
    model.eval()
    with torch.no_grad():
        policy, value = model(dummy_input)
    assert policy.shape == (BATCH_SIZE, NUM_MOVES), (
        f"{ModelClass.__name__} policy shape: expected ({BATCH_SIZE}, {NUM_MOVES}), "
        f"got {tuple(policy.shape)}"
    )
    assert value.shape == (BATCH_SIZE, 1), (
        f"{ModelClass.__name__} value shape: expected ({BATCH_SIZE}, 1), "
        f"got {tuple(value.shape)}"
    )


@pytest.mark.parametrize("ModelClass", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_parameter_budget(ModelClass):
    """All models must fit within the 100 k parameter budget."""
    model = ModelClass()
    n_params = model.count_parameters()
    assert n_params <= BUDGET, (
        f"{ModelClass.__name__} has {n_params:,} parameters, "
        f"exceeding the {BUDGET:,} budget."
    )


@pytest.mark.parametrize("ModelClass", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_value_range(ModelClass, dummy_input):
    """Value head output should be in (-1, 1) due to tanh activation."""
    model = ModelClass()
    model.eval()
    with torch.no_grad():
        _, value = model(dummy_input)
    assert value.min().item() > -1.0 - 1e-5, "Value below -1"
    assert value.max().item() < 1.0 + 1e-5, "Value above +1"


@pytest.mark.parametrize("ModelClass", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
def test_backward_pass(ModelClass, dummy_input):
    """Backward pass should compute gradients for all parameters."""
    model = ModelClass()
    model.train()
    policy, value = model(dummy_input)
    loss = policy.mean() + value.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"{ModelClass.__name__}: parameter '{name}' has no gradient."
            )
