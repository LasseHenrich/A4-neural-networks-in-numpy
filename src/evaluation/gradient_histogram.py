"""Gradient histogram metric."""

import numpy as np

from evaluation.metric import Metric
from nn.module import Module


class GradientHistogram(Metric):
    """Histogram of gradient values accumulated across all parameters.

    `compute` returns an integer array of bin counts with length
    `bins`. Call `np.histogram` on the result yourself if you need
    the bin edges as well.
    """

    name = "grad_histogram"

    def __init__(self, model: Module, bins: int = 50) -> None:
        self._model = model
        self._bins = bins
        self.reset()

    def reset(self) -> None:
        self._values: list[np.ndarray] = []

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        for _, grad in self._model.parameters():
            self._values.append(grad.ravel())

    def compute(self) -> np.ndarray:
        if not self._values:
            return np.zeros(self._bins, dtype=np.int64)
        all_grads = np.concatenate(self._values)
        counts, _ = np.histogram(all_grads, bins=self._bins)
        return counts.astype(np.int64)
