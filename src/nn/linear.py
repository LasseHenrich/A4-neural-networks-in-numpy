"""
Fully-connected (Linear) layer.

For a batch X of shape (B, in_features), the layer computes:

    Y = X @ W + b

where W has shape (in_features, out_features) and b has shape
(out_features,). The bias is broadcast across the batch.

Backward pass. Given the upstream gradient dY (same shape as Y):

    dW = X.T @ dY                # (in_features, out_features)
    db = dY.sum(axis=0)          # (out_features,)
    dX = dY @ W.T                # (B, in_features)

Initialization. Weights use He initialization, which keeps the variance
of activations roughly constant across layers when paired with ReLU:

    W ~ N(0, 2 / in_features)
    b = 0

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from collections.abc import Iterable

import numpy as np

from .module import Module


class Linear(Module):
    """
    A fully-connected layer:  y = x @ W + b.

    Attributes set in __init__:
        W  -- (in_features, out_features) weight matrix
        b  -- (out_features,) bias vector
        dW -- gradient of the loss w.r.t. W; same shape as W
        db -- gradient of the loss w.r.t. b; same shape as b

    Attributes set during forward (used by backward):
        x  -- the input passed to the most recent forward call
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int | None = None,
    ) -> None:
        """
        Initializes weights with He initialization and biases with zeros.

        Arguments:
            in_features  -- number of input features
            out_features -- number of output features
            seed         -- optional seed for reproducible initialization
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes y = x @ W + b and caches x for use in `backward`.

        Arguments:
            x -- (B, in_features) input batch

        Returns:
            (B, out_features) output batch
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Computes parameter gradients (stored in self.dW, self.db) and
        returns the gradient with respect to the layer's input.

        Must be called after `forward`; uses the cached input.

        Arguments:
            dout -- (B, out_features) gradient of the loss w.r.t. the
                    layer's output

        Returns:
            (B, in_features) gradient of the loss w.r.t. the layer's
            input
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def parameters(
        self,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Yields the layer's `(parameter, gradient)` pairs in the order
        `(W, dW)` then `(b, db)`. Consumed by `step`.
        """
        yield self.W, self.dW
        yield self.b, self.db
