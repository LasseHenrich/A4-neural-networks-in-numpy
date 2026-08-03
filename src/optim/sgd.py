"""
Stochastic gradient descent optimizer.

Three update rules are available via `momentum` and `nesterov`:

Vanilla SGD (`momentum=0`):
    param -= lr * grad

SGD with momentum (`momentum=β > 0`, `nesterov=False`):
    v = β * v + grad
    param -= lr * v

SGD with Nesterov momentum (`momentum=β > 0`, `nesterov=True`):
    v = β * v + grad
    param -= lr * (grad + β * v)
"""

from collections.abc import Iterable

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent with optional momentum and Nesterov
    momentum.

    Constructor arguments:
        parameters -- iterable of `(parameter, gradient)` array pairs,
                      typically `model.parameters()`. Both arrays in
                      each pair must be mutated in place by `backward`
                      (which is how `Linear.backward` writes `dW` and
                      `db`), so the optimizer keeps stable references.
        lr         -- learning rate
        momentum   -- momentum coefficient β ∈ [0, 1). 0 disables
                      momentum entirely (vanilla SGD).
        nesterov   -- if True, use Nesterov momentum instead of
                      standard momentum. Requires `momentum > 0`.
        weight_decay -- L2 penalty coefficient. When non-zero,
                      `weight_decay * param` is added to the gradient
                      before the momentum update (coupled L2, matching
                      `torch.optim.SGD`).
    """

    def __init__(
        self,
        parameters: Iterable[tuple[np.ndarray, np.ndarray]],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov=True requires momentum > 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocities: list[np.ndarray] = [
            np.zeros_like(param) for param, _ in self.parameters
        ]

    def step(self) -> None:
        """Applies one update step to every parameter."""
        # ------ WRITE YOUR CODE HERE ------
        pass
