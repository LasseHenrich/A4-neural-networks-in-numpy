"""
Loss functions: cross-entropy (with built-in softmax) and mean
squared error.

CrossEntropy is used for classification; MSE for regression.

Cross-entropy with built-in softmax
-----------------------------------
For a batch of B samples, given:
    logits z of shape (B, C)
    integer class labels y of shape (B,)

The softmax probabilities are:
    p_{i,c} = exp(z_{i,c}) / sum_k exp(z_{i,k})

The mean cross-entropy loss is:
    L = - (1 / B) * sum_i log(p_{i, y_i})

The combined gradient of L with respect to the logits z simplifies to:
    dL/dz_{i,c} = (p_{i,c} - 1[c == y_i]) / B

This combined form is numerically stable and avoids ever materializing
a separate softmax-then-log step.

To prevent overflow when any z is large, we subtract the per-row max
before exponentiating. This does not change the result of softmax.

Mean squared error
------------------
For predictions and targets of the same shape, MSE is the mean of
the element-wise squared differences:

    L = (1 / N) * sum_i (pred_i - target_i)^2

where N is the total number of elements. Unlike CrossEntropy it has
no softmax and expects real-valued targets rather than class
indices. Its gradient with respect to the predictions is:

    dL/dpred_i = (2 / N) * (pred_i - target_i)

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import numpy as np

from .module import Module


class CrossEntropy(Module):
    """
    Combined softmax + cross-entropy loss.

    Mirrors PyTorch's `nn.CrossEntropyLoss`: takes raw logits and
    integer class indices (not one-hot vectors), returns a scalar mean
    loss.

    Caches the softmax probabilities and class labels during forward
    so backward can compute (p - one_hot(y)) / B without recomputing.

    Note: `backward()` takes no upstream gradient — as a loss, it is
    where backpropagation begins and derives the initial gradient
    directly from its cached forward-pass state.
    """

    def __init__(self) -> None:
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the mean cross-entropy loss over the batch.

        Arguments:
            logits -- (B, C) raw scores from the network
            y      -- (B,) integer class labels in [0, C)

        Returns:
            float -- mean cross-entropy loss across the batch
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self) -> np.ndarray:
        """
        Returns the gradient of the loss with respect to the logits
        passed into the most recent `forward` call:

            dL/dz = (softmax(z) - one_hot(y)) / B

        Returns:
            (B, C) gradient with respect to the input logits
        """
        # ------ WRITE YOUR CODE HERE ------
        pass


class MSE(Module):
    """
    Mean squared error loss.

    Mirrors PyTorch's `nn.MSELoss` with the default `reduction="mean"`:
    takes a prediction array and a target array of the same shape and
    returns the mean of the element-wise squared differences as a
    scalar.

    Has no trainable parameters and applies no softmax. Caches the
    prediction-minus-target difference during forward so backward can
    compute (2 / N) * (pred - target) without recomputing.

    Note: `backward()` takes no upstream gradient — as a loss, it is
    where backpropagation begins and derives the initial gradient
    directly from its cached forward-pass state.
    """

    def __init__(self) -> None:
        # ------ WRITE YOUR CODE HERE ------
        pass

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the mean squared error over every element.

        Arguments:
            pred   -- predicted values of any shape
            target -- target values, same shape as pred

        Returns:
            float -- mean squared error across all elements
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def backward(self) -> np.ndarray:
        """
        Returns the gradient of the loss with respect to the prediction
        passed into the most recent `forward` call:

            dL/dpred = (2 / N) * (pred - target)

        where N is the total number of elements in pred.

        Returns:
            gradient with respect to the input prediction, same shape
            as pred
        """
        # ------ WRITE YOUR CODE HERE ------
        pass
