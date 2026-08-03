"""Gradient norm metric."""

import math

import numpy as np

from evaluation.metric import Metric
from nn.module import Module


class GradientNorm(Metric):
    """Global L2 gradient norm across all model parameters.

    Reports the mean per-batch global gradient norm over the epoch.
    Each call to ``update`` contributes one scalar
    ``sqrt(sum_params sum(grad**2))``; ``compute`` returns the mean of
    those scalars, or ``nan`` if no batches have been recorded.
    """

    name = "grad_norm"

    def __init__(self, model: Module) -> None:
        self._model = model
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        sq_sum = 0.0
        for _, grad in self._model.parameters():
            sq_sum += float(np.sum(grad**2))
        self._sum += math.sqrt(sq_sum)
        self._count += 1

    def compute(self) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum / self._count
