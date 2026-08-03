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
import math

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
        super().__init__()
        
        sigma = math.sqrt(2/in_features)
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0, scale=sigma, size=(in_features, out_features))
        self.b = np.zeros(shape=(out_features,))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes y = x @ W + b and caches x for use in `backward`.

        Arguments:
            x -- (B, in_features) input batch

        Returns:
            (B, out_features) output batch
        """
    
        self.x = x
        
        return x @ self.W + self.b
        

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
        if self.x is None:
            raise RuntimeError("`forward` has not been called yet")
        
        self.dW += self.x.T @ dout
        self.db += np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

    def parameters(
        self,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Yields the layer's `(parameter, gradient)` pairs in the order
        `(W, dW)` then `(b, db)`. Consumed by `step`.
        """
        yield self.W, self.dW
        yield self.b, self.db
