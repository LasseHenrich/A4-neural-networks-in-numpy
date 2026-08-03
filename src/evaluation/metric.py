"""Base metric classes."""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """
    A single stateful metric accumulated over a pass.

    `update` folds one batch of (outputs, targets) into running
    statistics, `compute` returns the finalized scalar, and `reset`
    clears the statistics before the next pass. Subclasses set `name`
    (the key under which the value appears in results) and take any
    configuration they need in their own constructor.

    Dual contract
    -------------
    `update(outputs, targets)` serves two distinct roles depending on
    the subclass:

    * **Performance metrics** (`Accuracy`, `ConfusionMatrix`) use
      both arguments to measure how well predictions match targets.

    * **Diagnostic metrics** (`GradientNorm`, `WeightNorm`,
      `GradWeightRatio`, `GradientHistogram`, `UpdateMagnitude`)
      intentionally ignore `outputs` and `targets`. They instead read
      gradient and weight arrays directly from the model captured at
      construction time. Because they inspect gradient buffers, they
      must be updated *immediately after* `backward()` and *before*
      `zero_grad()` — calling them after `zero_grad()` would see all
      zeros.
    """

    name: str

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None: ...

    @abstractmethod
    def compute(self) -> float | np.ndarray: ...


class MetricCollection:
    """
    Holds a list of `Metric` objects and drives them together.

    `update` forwards each batch to every metric, `compute` gathers a
    `{name: value}` dict from them, and `reset` clears them all.
    Adding a metric is just another object in the list — the training
    loop is unaffected.
    """

    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        for metric in self.metrics:
            metric.update(outputs, targets)

    def compute(self) -> dict[str, float | np.ndarray]:
        return {metric.name: metric.compute() for metric in self.metrics}
