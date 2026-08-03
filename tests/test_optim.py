import numpy as np
import pytest

from nn import SGD, Adam


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_sgd_step_smoke():
    p = np.zeros(3)
    g = np.ones(3)
    result = SGD([(p, g)], lr=0.1).step()
    assert result is None


def test_adam_step_smoke():
    p = np.zeros(3)
    g = np.ones(3)
    result = Adam([(p, g)]).step()
    assert result is None


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------


def test_sgd_vanilla_update():
    pass


def test_sgd_updates_all_parameters():
    pass


def test_sgd_momentum_accumulates_velocity():
    pass


def test_sgd_nesterov_update():
    pass


def test_sgd_nesterov_requires_momentum():
    pass


def test_sgd_weight_decay_update():
    pass


def test_sgd_weight_decay_preserves_grad_buffer():
    pass


def test_sgd_negative_weight_decay_raises():
    pass


def test_sgd_zero_grad_clears_buffer():
    pass


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


def test_adam_first_step_is_bias_corrected():
    pass


def test_adam_converges_to_minimum():
    # Minimize (x - 3)^2 ; grad = 2 (x - 3).
    pass


def test_adam_weight_decay_shrinks_param():
    pass


def test_adam_weight_decay_preserves_grad_buffer():
    pass


def test_adam_negative_weight_decay_raises():
    pass


def test_adam_zero_grad_clears_buffer():
    pass
