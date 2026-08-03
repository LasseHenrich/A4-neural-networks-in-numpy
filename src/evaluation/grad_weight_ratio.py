"""Gradient-to-weight norm ratio metric."""

import math

import numpy as np

from evaluation.metric import Metric
from nn.module import Module


class GradWeightRatio(Metric):
    """Ratio of global gradient norm to global weight norm.

    Reports the mean per-batch ratio over the epoch.  Each call to
    ``update`` computes ``sqrt(sum grad**2) / sqrt(sum param**2)`` for
    that batch; batches where the weight norm is zero are skipped.
    ``compute`` returns the mean of the recorded ratios, or ``nan`` if
    no valid batches have been recorded.

    A healthy value is roughly 0.01. Much smaller suggests learning has
    stalled; much larger suggests an unstable update step.
    """

    name = "grad_weight_ratio"

    def __init__(self, model: Module) -> None:
        self._model = model
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        grad_sq = 0.0
        weight_sq = 0.0
        for param, grad in self._model.parameters():
            grad_sq += float(np.sum(grad**2))
            weight_sq += float(np.sum(param**2))
        weight_norm = math.sqrt(weight_sq)
        if weight_norm == 0.0:
            return
        self._sum += math.sqrt(grad_sq) / weight_norm
        self._count += 1

    def compute(self) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum / self._count
