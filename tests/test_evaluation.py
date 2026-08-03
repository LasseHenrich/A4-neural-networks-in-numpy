import numpy as np
import pytest

from evaluation.accuracy import Accuracy, accuracy
from evaluation.metric import Metric, MetricCollection


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_accuracy_smoke():
    result = accuracy(np.array([0, 1]), np.array([0, 1]))
    assert isinstance(result, float)


def test_accuracy_metric_smoke():
    m = Accuracy()
    outputs = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([0, 1])
    m.update(outputs, targets)
    assert isinstance(m.compute(), float)


def test_metric_collection_smoke():
    col = MetricCollection([Accuracy()])
    outputs = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([0, 1])
    col.update(outputs, targets)
    result = col.compute()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# accuracy (array-based)
# ---------------------------------------------------------------------------


def test_accuracy_perfect_predictions():
    pass


def test_accuracy_all_wrong():
    pass


def test_accuracy_half_correct():
    pass


def test_accuracy_shape_mismatch_raises():
    pass


def test_accuracy_empty_raises():
    pass


# ---------------------------------------------------------------------------
# Metric (ABC)
# ---------------------------------------------------------------------------


def test_metric_is_abstract():
    pass


# ---------------------------------------------------------------------------
# Accuracy (Metric)
# ---------------------------------------------------------------------------


def test_accuracy_metric_name():
    pass


def test_accuracy_metric_perfect():
    pass


def test_accuracy_metric_all_wrong():
    pass


def test_accuracy_metric_half_correct():
    pass


def test_accuracy_metric_empty_returns_nan():
    pass


def test_accuracy_metric_reset_clears_state():
    pass


def test_accuracy_metric_accumulates_across_batches():
    pass


def test_accuracy_metric_weighted_batches():
    pass


# ---------------------------------------------------------------------------
# MetricCollection
# ---------------------------------------------------------------------------


def test_metric_collection_compute_keys():
    pass


def test_metric_collection_compute_value():
    pass


def test_metric_collection_reset():
    pass


def test_metric_collection_multiple_metrics():
    pass


def test_metric_collection_empty():
    pass
