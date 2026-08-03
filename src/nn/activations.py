"""
Non-linear activation functions.

ReLU (Rectified Linear Unit) is the most common activation function
used between hidden layers:

    ReLU(x) = max(0, x)

Its derivative is 1 for inputs strictly greater than 0, and 0
otherwise. The convention at exactly x = 0 is 0, which is what
`x > 0` returns when broadcast.

Leaky ReLU avoids the "dying ReLU" problem by letting a small,
fixed slope through for non-positive inputs:

    LeakyReLU(x) = x          if x > 0
                 = alpha * x  if x <= 0

Its derivative is 1 where the input was positive and `alpha`
elsewhere.

GELU (Gaussian Error Linear Unit) is a smooth activation common in
modern transformer architectures. We use the tanh approximation
(matching PyTorch's approximate="tanh" mode), which stays in pure
NumPy:

    GELU(x) = 0.5 * x * (1 + tanh(c * (x + a * x**3)))

with c = sqrt(2 / pi) and a = 0.044715.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np

from .module import Module


class ReLU(Module):
    """
    Element-wise ReLU activation:  y = max(0, x).

    Holds no trainable parameters. Caches the boolean mask
    (x > 0) during the forward pass for use in the backward pass.
    """

    def __init__(self) -> None:
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies ReLU element-wise and caches the positive-input mask.

        Arguments:
            x -- input array of any shape

        Returns:
            array of the same shape as x with negatives clipped to 0
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Multiplies dout by the cached mask. Inputs that were <= 0 in
        the forward pass contribute zero to the gradient.

        Arguments:
            dout -- gradient of the loss w.r.t. the layer's output;
                    same shape as the cached input

        Returns:
            gradient of the loss w.r.t. the layer's input
        """
        # ------ WRITE YOUR CODE HERE ------
        pass


class LeakyReLU(Module):
    """
    Element-wise leaky ReLU activation:

        y = x          if x > 0
        y = alpha * x  otherwise

    Holds no trainable parameters. Caches the boolean mask
    (x > 0) during the forward pass for use in the backward pass.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies leaky ReLU element-wise and caches the positive mask.

        Arguments:
            x -- input array of any shape

        Returns:
            array of the same shape as x with non-positive entries
            scaled by the negative slope
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Multiplies dout by 1 where the input was positive and by the
        negative slope where it was not.

        Arguments:
            dout -- gradient of the loss w.r.t. the layer's output;
                    same shape as the cached input

        Returns:
            gradient of the loss w.r.t. the layer's input
        """
        # ------ WRITE YOUR CODE HERE ------
        pass


class GELU(Module):
    """
    Element-wise GELU activation (tanh approximation):

        y = 0.5 * x * (1 + tanh(C * (x + A * x**3)))

    Holds no trainable parameters. Caches the input during the
    forward pass because the gradient depends on x directly.
    """

    C = float(np.sqrt(2.0 / np.pi))
    A = 0.044715

    def __init__(self) -> None:
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies GELU element-wise and caches the input.

        Arguments:
            x -- input array of any shape

        Returns:
            array of the same shape as x
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Computes the gradient using the cached input via the product
        and chain rules on the tanh approximation.

        Arguments:
            dout -- gradient of the loss w.r.t. the layer's output;
                    same shape as the cached input

        Returns:
            gradient of the loss w.r.t. the layer's input
        """
        # ------ WRITE YOUR CODE HERE ------
        pass
