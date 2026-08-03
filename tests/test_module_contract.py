"""
Tests that every concrete Module subclass properly initializes
``self.training`` by calling ``super().__init__()``.
"""

import pytest

from nn.activations import GELU, LeakyReLU, ReLU
from nn.dropout import Dropout
from nn.linear import Linear
from nn.loss import CrossEntropy, MSE
from nn.mlp import MLP
from nn.sequential import Sequential


# ---------------------------------------------------------------------------
# Parametrized contract check
# ---------------------------------------------------------------------------

_SUBJECTS = [
    Linear(2, 3),
    ReLU(),
    LeakyReLU(),
    GELU(),
    CrossEntropy(),
    MSE(),
    Dropout(0.5),
    Sequential([ReLU()]),
    MLP([2, 3]),
]

_IDS = [
    "Linear",
    "ReLU",
    "LeakyReLU",
    "GELU",
    "CrossEntropy",
    "MSE",
    "Dropout",
    "Sequential",
    "MLP",
]


@pytest.mark.parametrize("obj", _SUBJECTS, ids=_IDS)
def test_training_is_instance_variable(obj: object) -> None:
    """training must live in the instance dict, not just the class."""
    assert "training" in vars(obj), (
        f"{type(obj).__name__} did not initialize self.training — "
        "did you call super().__init__() in __init__?"
    )


@pytest.mark.parametrize("obj", _SUBJECTS, ids=_IDS)
def test_training_defaults_to_true(obj: object) -> None:
    """training must be True immediately after construction."""
    assert obj.training is True, (  # type: ignore[union-attr]
        f"{type(obj).__name__} did not initialize self.training — "
        "did you call super().__init__() in __init__?"
    )
