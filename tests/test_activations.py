import numpy as np
import pytest

from nn.activations import GELU, LeakyReLU, ReLU


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_relu_forward_smoke():
    relu = ReLU()
    out = relu.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_relu_backward_smoke():
    relu = ReLU()
    relu.forward(np.array([[1.0, -1.0]]))
    grad = relu.backward(np.array([[1.0, 1.0]]))
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (1, 2)


# ---------------------------------------------------------------------------
# ReLU.forward
# ---------------------------------------------------------------------------


def test_relu_forward_clips_negatives_to_zero():
    pass


def test_relu_forward_preserves_shape():
    pass


# ---------------------------------------------------------------------------
# ReLU.backward
# ---------------------------------------------------------------------------


def test_relu_backward_zeros_negative_inputs():
    pass


def test_relu_backward_zero_at_zero_input():
    pass


def test_relu_backward_passes_positive_gradients():
    pass


def test_relu_backward_without_forward_raises():
    pass


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------


def test_leaky_relu_forward_smoke():
    leaky = LeakyReLU()
    out = leaky.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_leaky_relu_forward_scales_negatives():
    pass


def test_leaky_relu_backward_uses_slope_for_negatives():
    pass


def test_leaky_relu_backward_without_forward_raises():
    pass


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------


def test_gelu_forward_smoke():
    gelu = GELU()
    out = gelu.forward(np.array([[1.0, -1.0]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_gelu_forward_zero_maps_to_zero():
    pass


def test_gelu_forward_saturates_to_identity_and_zero():
    pass


def test_gelu_backward_matches_finite_differences():
    pass


def test_gelu_backward_without_forward_raises():
    pass
