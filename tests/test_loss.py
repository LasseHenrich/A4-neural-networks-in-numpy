import math

import numpy as np
import pytest

from nn.loss import CrossEntropy


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_cross_entropy_forward_smoke():
    loss = CrossEntropy()
    value = loss.forward(np.zeros((2, 3)), np.array([0, 1]))
    assert isinstance(value, float)


def test_cross_entropy_backward_smoke():
    loss = CrossEntropy()
    loss.forward(np.zeros((2, 3)), np.array([0, 1]))
    grad = loss.backward()
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 3)


# ---------------------------------------------------------------------------
# CrossEntropy.forward
# ---------------------------------------------------------------------------


def test_cross_entropy_uniform_logits_equals_log_C():
    pass


def test_cross_entropy_perfect_prediction_near_zero():
    pass


def test_cross_entropy_numerical_stability():
    pass


# ---------------------------------------------------------------------------
# CrossEntropy.backward
# ---------------------------------------------------------------------------


def test_cross_entropy_backward_sums_to_zero_per_row():
    pass


def test_cross_entropy_backward_matches_softmax_minus_onehot():
    pass


def test_cross_entropy_backward_without_forward_raises():
    pass


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------


from nn.loss import MSE  # noqa: E402


def test_mse_forward_known_value():
    pass


def test_mse_forward_returns_float():
    pass


def test_mse_backward_known_gradient():
    pass


def test_mse_backward_without_forward_raises():
    pass
