"""Weight norm metric."""

import math

import numpy as np

from evaluation.metric import Metric
from nn.module import Module


class WeightNorm(Metric):
    """Global L2 weight norm across all model parameters.

    Reports the mean per-batch global weight norm over the epoch.
    Each call to ``update`` contributes one scalar
    ``sqrt(sum_params sum(param**2))``; ``compute`` returns the mean of
    those scalars, or ``nan`` if no batches have been recorded.
    """

    name = "weight_norm"

    def __init__(self, model: Module) -> None:
        self._model = model
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        sq_sum = 0.0
        for param, _ in self._model.parameters():
            sq_sum += float(np.sum(param**2))
        self._sum += math.sqrt(sq_sum)
        self._count += 1

    def compute(self) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum / self._count
