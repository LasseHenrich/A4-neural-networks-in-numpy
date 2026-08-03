"""
Dropout regularization layer.

During training, dropout randomly zeroes each element of the input with
probability `p`, then scales the surviving elements by `1 / (1 - p)`.
This "inverted dropout" convention (matching PyTorch's `nn.Dropout`)
keeps the expected value of each activation unchanged, so no rescaling
is needed at evaluation time:

    keep_prob = 1 - p
    mask      = (uniform(0, 1) < keep_prob) / keep_prob   # 0 or 1/keep_prob
    y         = x * mask        (training)
    y         = x               (evaluation)

Because the surviving units are already scaled up during training, the
forward pass is the identity in evaluation mode. The same mask is reused
in the backward pass: gradients flow only through the units that were
kept, scaled by the same `1 / keep_prob` factor.

Dropout holds no trainable parameters. Its behaviour depends on the
`training` flag inherited from `Module`, which the training loop toggles
via `train()` / `eval()`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np

from .module import Module


class Dropout(Module):
    """
    Inverted-dropout layer:  randomly drops units during training and is
    the identity during evaluation.

    Attributes set in __init__:
        p    -- probability of dropping each unit, in [0, 1)
        rng  -- NumPy generator used to sample the dropout mask
        mask -- scaled keep mask cached during the most recent training
                forward pass; None until forward is called in training
                mode
    """

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        """
        Stores the drop probability and a seeded RNG for reproducible
        masks.

        Arguments:
            p    -- probability of dropping each unit; must satisfy
                    0 <= p < 1
            seed -- optional seed for reproducible mask sampling
        """
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability p must satisfy 0 <= p < 1, got {p}")
        
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies inverted dropout in training mode; returns the input
        unchanged in evaluation mode.

        Arguments:
            x -- input array of any shape

        Returns:
            array of the same shape as x; some entries zeroed and the
            rest scaled up by 1 / (1 - p) during training, or x itself
            during evaluation
        """
        if not self.training:
            return x
        
        q = 1 - self.p # keep probability
        self.mask = (self.rng.random(x.shape) < q) / q
        
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Routes the upstream gradient through the cached keep mask in
        training mode; passes it through unchanged in evaluation mode.

        Arguments:
            dout -- gradient of the loss w.r.t. the layer's output; same
                    shape as the input to the most recent forward call

        Returns:
            gradient of the loss w.r.t. the layer's input
        """
        if not self.training:
            return dout
        
        if self.mask is None:
            raise RuntimeError("`forward` has not been called yet")
        
        return dout * self.mask
        
