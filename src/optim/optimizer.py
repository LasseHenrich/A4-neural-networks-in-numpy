"""Base class for all optimizers."""

from collections.abc import Iterable

import numpy as np


class Optimizer:
    """
    Base class for parameter optimizers.

    Subclasses must override `step` to implement a concrete update rule.
    `zero_grad` is provided here and shared by all subclasses.

    The optimizer is initialized with the iterable returned by
    `model.parameters()` and exposes two methods used by the training
    loop:

        optimizer.zero_grad()  # clear all gradient buffers
        optimizer.step()       # apply one parameter update

    `zero_grad` clears the gradient buffers that `backward` accumulates
    into, and must be called before each backward pass to avoid carrying
    gradients over from previous batches.

    Constructor arguments:
        parameters -- iterable of `(parameter, gradient)` array pairs,
                      typically `model.parameters()`.
        lr         -- learning rate
    """

    def __init__(
        self,
        parameters: Iterable[tuple[np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        self.parameters: list[tuple[np.ndarray, np.ndarray]] = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Sets every gradient buffer to zero in place."""
        for _, grad in self.parameters:
            grad[...] = 0.0

    def step(self) -> None:
        """Applies one update step to every parameter."""
        raise NotImplementedError
