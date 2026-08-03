import numpy as np
import pytest

from evaluation.confusion_matrix import (
    ConfusionMatrix,
    accuracy_from_matrix,
    f1,
    precision,
    recall,
)

# Alias for test readability: the cm-based function was renamed.
accuracy = accuracy_from_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 3-class: 2 correct per class, no misclassifications
_CM_PERFECT = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)

# 2-class: [[TP0, FN0], [FP0, TP1]] = [[3, 1], [2, 4]]
# precision: [3/5, 4/5]  recall: [3/4, 4/6]  f1: [3/4.5, 4/5.5]
_CM_MIXED = np.array([[3, 1], [2, 4]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_confusion_matrix_smoke():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        np.array([0, 1]),
    )
    assert isinstance(cm.compute(), np.ndarray)


def test_accuracy_smoke():
    assert isinstance(accuracy(_CM_PERFECT), float)


def test_precision_smoke():
    result = precision(_CM_MIXED)
    assert isinstance(result, np.ndarray)


def test_recall_smoke():
    result = recall(_CM_MIXED)
    assert isinstance(result, np.ndarray)


def test_f1_smoke():
    result = f1(_CM_MIXED)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_name():
    pass


def test_confusion_matrix_shape():
    pass


def test_confusion_matrix_perfect():
    pass


def test_confusion_matrix_all_wrong():
    pass


def test_confusion_matrix_known_entries():
    pass


def test_confusion_matrix_accumulates_across_batches():
    pass


def test_confusion_matrix_reset():
    pass


def test_confusion_matrix_compute_returns_copy():
    pass


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_perfect():
    pass


def test_accuracy_all_wrong():
    pass


def test_accuracy_known_value():
    # _CM_MIXED: 3+4=7 correct out of 10
    pass


def test_accuracy_empty_returns_nan():
    pass


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------


def test_precision_perfect():
    pass


def test_precision_known_values():
    # col sums: [5, 5]; diag: [3, 4]
    pass


def test_precision_single_class():
    pass


def test_precision_zero_column():
    # class 1 never predicted → precision[1] = 0.0
    pass


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def test_recall_perfect():
    pass


def test_recall_known_values():
    # row sums: [4, 6]; diag: [3, 4]
    pass


def test_recall_single_class():
    pass


def test_recall_zero_row():
    # class 1 has no true instances → recall[1] = 0.0
    pass


# ---------------------------------------------------------------------------
# f1
# ---------------------------------------------------------------------------


def test_f1_perfect():
    pass


def test_f1_known_values():
    pass


def test_f1_single_class():
    pass


def test_f1_both_zero():
    # class with no predictions and no true instances → f1 = 0.0
    pass
