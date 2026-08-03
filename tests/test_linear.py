import numpy as np
import pytest

from nn import SGD
from nn.linear import Linear


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_linear_init_smoke():
    layer = Linear(4, 3, seed=0)
    assert layer.W.shape == (4, 3)
    assert layer.b.shape == (3,)


def test_linear_forward_smoke():
    layer = Linear(4, 3, seed=0)
    out = layer.forward(np.zeros((2, 4)))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_linear_backward_smoke():
    layer = Linear(4, 3, seed=0)
    layer.forward(np.zeros((2, 4)))
    grad = layer.backward(np.zeros((2, 3)))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 4)


# ---------------------------------------------------------------------------
# Linear.__init__
# ---------------------------------------------------------------------------


def test_linear_bias_initialized_to_zeros():
    pass


def test_linear_he_initialization_variance():
    pass


def test_linear_seed_reproducible():
    pass


def test_linear_different_seeds_differ():
    pass


# ---------------------------------------------------------------------------
# Linear.forward
# ---------------------------------------------------------------------------


def test_linear_forward_matches_manual_matmul():
    pass


def test_linear_forward_caches_input():
    pass


# ---------------------------------------------------------------------------
# Linear.backward
# ---------------------------------------------------------------------------


def test_linear_backward_dx_shape():
    pass


def test_linear_backward_computes_dW():
    pass


def test_linear_backward_computes_db():
    pass


def test_linear_backward_computes_dx():
    pass


# ---------------------------------------------------------------------------
# Linear.parameters
# ---------------------------------------------------------------------------


def test_linear_parameters_yields_W_and_b():
    pass


def test_linear_sgd_step_changes_weights():
    pass


def test_linear_sgd_step_applies_sgd_rule():
    pass
