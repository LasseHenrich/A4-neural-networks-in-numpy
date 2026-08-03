"""Accuracy metric and helper."""

import numpy as np

from evaluation.metric import Metric


class Accuracy(Metric):
    """Sample-weighted classification accuracy."""

    name = "acc"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total = 0.0
        self._n = 0

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        batch_n = len(targets)
        preds = outputs.argmax(axis=1)
        self._total += accuracy(preds, targets) * batch_n
        self._n += batch_n

    def compute(self) -> float:
        if self._n == 0:
            return float("nan")
        return self._total / self._n


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the fraction of predictions that match the true labels.

    Both arrays must contain integer class indices. Their shapes must
    match. An empty input raises ValueError.

    Arguments:
        y_pred -- (N,) predicted class labels
        y_true -- (N,) true class labels

    Returns:
        float -- accuracy in [0, 1]
    """
    # ------ WRITE YOUR CODE HERE ------
    pass
