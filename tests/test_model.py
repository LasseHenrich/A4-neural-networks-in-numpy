import numpy as np
import pytest

from nn import SGD
from nn.activations import GELU, LeakyReLU, ReLU
from nn.dropout import Dropout
from nn.linear import Linear
from nn.loss import CrossEntropy
from nn.mlp import MLP
from nn.sequential import Sequential


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 0) -> Sequential:
    return Sequential(
        [
            Linear(4, 5, seed=seed),
            ReLU(),
            Linear(5, 3, seed=seed + 1),
        ]
    )


def test_model_forward_smoke():
    out = _tiny_model().forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_model_backward_smoke():
    model = _tiny_model()
    model.forward(np.ones((2, 4)))
    grad = model.backward(np.ones((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Sequential.forward
# ---------------------------------------------------------------------------


def test_model_forward_matches_manual_pipeline():
    pass


def test_model_callable_equals_forward():
    pass


# ---------------------------------------------------------------------------
# Sequential.backward
# ---------------------------------------------------------------------------


def test_model_backward_runs_layers_in_reverse():
    pass


# ---------------------------------------------------------------------------
# Sequential.parameters
# ---------------------------------------------------------------------------


def test_model_parameters_chains_layer_parameters():
    pass


def test_model_parameters_skips_relu():
    pass


# ---------------------------------------------------------------------------
# SGD step via Sequential.parameters
# ---------------------------------------------------------------------------


def test_model_sgd_step_changes_every_linear():
    pass


# ---------------------------------------------------------------------------
# End-to-end overfit
# ---------------------------------------------------------------------------


def test_model_can_overfit_tiny_dataset():
    pass


# ---------------------------------------------------------------------------
# Sequential empty-layers guard
# ---------------------------------------------------------------------------


def test_sequential_empty_raises():
    pass


# ===========================================================================
# MLP tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_mlp_forward_smoke():
    mlp = MLP([4, 8, 3], activation="relu", layer_seed=0)
    out = mlp.forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_mlp_backward_smoke():
    mlp = MLP([4, 8, 3], activation="relu", layer_seed=0)
    mlp.forward(np.ones((2, 4)))
    grad = mlp.backward(np.ones((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


def test_mlp_layer_count_one_hidden():
    # [784, 128, 10]: Linear, ReLU, Linear → 3 layers
    pass


def test_mlp_layer_count_two_hidden():
    # [4, 8, 16, 3]: Linear, ReLU, Linear, ReLU, Linear → 5 layers
    pass


def test_mlp_linear_widths():
    pass


def test_mlp_activation_between_linears_not_after_last():
    pass


def test_mlp_leaky_relu_activation():
    pass


def test_mlp_gelu_activation():
    pass


def test_mlp_unknown_activation_raises():
    pass


# ---------------------------------------------------------------------------
# Dropout injection
# ---------------------------------------------------------------------------


def test_mlp_no_dropout_by_default():
    pass


def test_mlp_dropout_injected_after_hidden_activations():
    # [4, 8, 16, 3] with dropout: Lin, Act, Drop, Lin, Act, Drop, Lin
    pass


def test_mlp_dropout_layer_count():
    # Two hidden layers → two Dropout layers
    pass


# ---------------------------------------------------------------------------
# Train / eval mode propagation
# ---------------------------------------------------------------------------


def test_mlp_train_propagates_to_children():
    pass


def test_mlp_eval_propagates_to_children():
    pass
