"""
Module base class.

Every learnable component in this library inherits from `Module`. The
class is intentionally tiny and mirrors the calling conventions of
PyTorch's `nn.Module`:

- `module(x)` is short for `module.forward(x)`.
- `module.parameters()` yields `(param, grad)` pairs that the
  optimizer consumes to perform parameter updates.
- Subclasses override `forward` and `backward`. Subclasses that hold
  trainable weights also override `parameters`.

This file is provided. You should not need to edit it.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np


class Module:
    """
    Base class for every layer, loss, and model in `nn`.

    Subclasses implement `forward` and `backward`. Subclasses that own
    trainable parameters also implement `parameters` to expose them.

    A `training` flag mirrors PyTorch's `nn.Module.training`. It is
    initialized per-instance in `__init__`, so every subclass
    constructor must call `super().__init__()` to ensure the flag is
    set as an instance variable rather than falling back to a class
    attribute. Modules with train/eval behaviour (e.g. `Dropout`) read
    `self.training` during their forward and backward passes.

    Backward-pass asymmetry
    -----------------------
    Layer `backward(dout)` receives an upstream gradient and transforms
    it, passing the result further back through the network. Loss
    `backward()` takes *no* argument — the loss is where backpropagation
    begins, so it derives the initial gradient directly from its cached
    forward-pass state and returns it to the caller.
    """

    training: bool

    def __init__(self) -> None:
        self.training = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forwards the call to `self.forward`, mirroring PyTorch."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def train(self) -> None:
        """Sets this module to training mode (`training = True`)."""
        self.training = True

    def eval(self) -> None:
        """Sets this module to evaluation mode (`training = False`)."""
        self.training = False

    def parameters(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Yields `(parameter_array, gradient_array)` pairs.

        The default implementation returns an empty iterator, which is
        the correct behaviour for layers without trainable weights
        (e.g. `ReLU`, `CrossEntropy`).
        """
        return iter(())
