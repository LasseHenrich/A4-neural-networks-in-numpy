import numpy as np
import pytest

from evaluation.grad_weight_ratio import GradWeightRatio
from evaluation.gradient_histogram import GradientHistogram
from evaluation.gradient_norm import GradientNorm
from evaluation.metric import Metric
from evaluation.update_magnitude import UpdateMagnitude
from evaluation.weight_norm import WeightNorm
from nn.module import Module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(params: list[np.ndarray], grads: list[np.ndarray]) -> Module:
    class _M(Module):
        def parameters(self):  # type: ignore[override]
            return zip(params, grads)

    return _M()


# param norm = 5, grad norm = 2.5, ratio = 0.5
_PARAM = np.array([3.0, 4.0])
_GRAD = np.array([1.5, 2.0])


def _single_layer_model() -> Module:
    return _make_model([_PARAM.copy()], [_GRAD.copy()])


def _empty_model() -> Module:
    return _make_model([], [])


# ---------------------------------------------------------------------------
# GradientNorm
# ---------------------------------------------------------------------------


def test_gradient_norm_is_metric():
    assert isinstance(GradientNorm(_single_layer_model()), Metric)


def test_gradient_norm_name():
    assert GradientNorm.name == "grad_norm"


def test_gradient_norm_nan_before_update():
    m = GradientNorm(_single_layer_model())
    assert np.isnan(m.compute())


def test_gradient_norm_single_layer():
    m = GradientNorm(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(2.5)


def test_gradient_norm_two_layers_global():
    # Squared contributions must be summed across layers before the
    # sqrt, not normed per-layer and then summed.
    model = _make_model(
        [np.array([3.0, 0.0]), np.array([0.0])],
        [np.array([3.0, 0.0]), np.array([4.0])],
    )
    m = GradientNorm(model)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(5.0)


def test_gradient_norm_accumulates_across_updates():
    # Each update() call is one batch; the metric returns the mean of
    # the per-batch norms over the epoch.  Two batches with identical
    # gradients give mean = single-batch norm.
    m = GradientNorm(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(2.5)


def test_gradient_norm_reset():
    m = GradientNorm(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.reset()
    assert np.isnan(m.compute())


def test_gradient_norm_outputs_targets_ignored():
    # update() reads parameters from the model, not from its arguments.
    m = GradientNorm(_single_layer_model())
    m.update(np.random.rand(5, 3), np.array([9, 9, 9, 9, 9]))
    assert m.compute() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# WeightNorm
# ---------------------------------------------------------------------------


def test_weight_norm_is_metric():
    assert isinstance(WeightNorm(_single_layer_model()), Metric)


def test_weight_norm_name():
    assert WeightNorm.name == "weight_norm"


def test_weight_norm_nan_before_update():
    m = WeightNorm(_single_layer_model())
    assert np.isnan(m.compute())


def test_weight_norm_single_layer():
    m = WeightNorm(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(5.0)


def test_weight_norm_ignores_grad():
    model = _make_model(
        [np.array([3.0, 4.0])],
        [np.zeros(2)],  # grad is zero; norm should still reflect params
    )
    m = WeightNorm(model)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(5.0)


def test_weight_norm_two_layers_global():
    model = _make_model(
        [np.array([3.0, 0.0]), np.array([0.0, 4.0])],
        [np.zeros(2), np.zeros(2)],
    )
    m = WeightNorm(model)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(5.0)


def test_weight_norm_reset():
    m = WeightNorm(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.reset()
    assert np.isnan(m.compute())


# ---------------------------------------------------------------------------
# GradWeightRatio
# ---------------------------------------------------------------------------


def test_grad_weight_ratio_is_metric():
    assert isinstance(GradWeightRatio(_single_layer_model()), Metric)


def test_grad_weight_ratio_name():
    assert GradWeightRatio.name == "grad_weight_ratio"


def test_grad_weight_ratio_nan_before_update():
    m = GradWeightRatio(_single_layer_model())
    assert np.isnan(m.compute())


def test_grad_weight_ratio_known_value():
    # param norm = 5, grad norm = 2.5, ratio = 0.5
    m = GradWeightRatio(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(0.5)


def test_grad_weight_ratio_zero_weight_nan():
    model = _make_model(
        [np.zeros(3)],
        [np.array([1.0, 2.0, 3.0])],
    )
    m = GradWeightRatio(model)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert np.isnan(m.compute())


def test_grad_weight_ratio_reset():
    m = GradWeightRatio(_single_layer_model())
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.reset()
    assert np.isnan(m.compute())


# ---------------------------------------------------------------------------
# GradientHistogram
# ---------------------------------------------------------------------------


def test_gradient_histogram_is_metric():
    assert isinstance(GradientHistogram(_single_layer_model()), Metric)


def test_gradient_histogram_name():
    assert GradientHistogram.name == "grad_histogram"


def test_gradient_histogram_empty_returns_zeros():
    m = GradientHistogram(_empty_model(), bins=10)
    result = m.compute()
    np.testing.assert_array_equal(result, np.zeros(10, dtype=np.int64))


def test_gradient_histogram_shape():
    m = GradientHistogram(_single_layer_model(), bins=20)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute().shape == (20,)


def test_gradient_histogram_dtype():
    m = GradientHistogram(_single_layer_model(), bins=10)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute().dtype == np.int64


def test_gradient_histogram_count_equals_elements():
    # Every gradient element must land in exactly one bin; testing the
    # total avoids brittle assertions about individual bin boundaries.
    m = GradientHistogram(_single_layer_model(), bins=5)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute().sum() == _GRAD.size


def test_gradient_histogram_accumulates_across_layers():
    # two-layer model with 2 elements each → 4 total counts
    model = _make_model(
        [np.zeros(2), np.zeros(2)],
        [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
    )
    m = GradientHistogram(model, bins=5)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute().sum() == 4


def test_gradient_histogram_bins_param():
    m5 = GradientHistogram(_single_layer_model(), bins=5)
    m10 = GradientHistogram(_single_layer_model(), bins=10)
    m5.update(np.zeros((1, 1)), np.zeros(1))
    m10.update(np.zeros((1, 1)), np.zeros(1))
    assert m5.compute().shape == (5,)
    assert m10.compute().shape == (10,)


def test_gradient_histogram_reset():
    m = GradientHistogram(_single_layer_model(), bins=5)
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.reset()
    np.testing.assert_array_equal(m.compute(), np.zeros(5, dtype=np.int64))


# ---------------------------------------------------------------------------
# UpdateMagnitude
# ---------------------------------------------------------------------------


def test_update_magnitude_is_metric():
    assert isinstance(UpdateMagnitude(_single_layer_model(), lr=0.1), Metric)


def test_update_magnitude_name():
    assert UpdateMagnitude.name == "update_magnitude"


def test_update_magnitude_nan_before_update():
    m = UpdateMagnitude(_single_layer_model(), lr=0.1)
    assert np.isnan(m.compute())


def test_update_magnitude_known_value():
    # lr=0.1, grad_norm=2.5, weight_norm=5.0 → 0.1 * 2.5/5.0 = 0.05
    m = UpdateMagnitude(_single_layer_model(), lr=0.1)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert m.compute() == pytest.approx(0.05)


def test_update_magnitude_lr_scaling():
    # Checks linearity in lr without a second hardcoded absolute value.
    m1 = UpdateMagnitude(_single_layer_model(), lr=0.1)
    m2 = UpdateMagnitude(_single_layer_model(), lr=0.2)
    m1.update(np.zeros((1, 1)), np.zeros(1))
    m2.update(np.zeros((1, 1)), np.zeros(1))
    assert m2.compute() == pytest.approx(2 * m1.compute())


def test_update_magnitude_zero_weight_nan():
    model = _make_model(
        [np.zeros(3)],
        [np.array([1.0, 2.0, 3.0])],
    )
    m = UpdateMagnitude(model, lr=0.1)
    m.update(np.zeros((1, 1)), np.zeros(1))
    assert np.isnan(m.compute())


def test_update_magnitude_reset():
    m = UpdateMagnitude(_single_layer_model(), lr=0.1)
    m.update(np.zeros((1, 1)), np.zeros(1))
    m.reset()
    assert np.isnan(m.compute())
