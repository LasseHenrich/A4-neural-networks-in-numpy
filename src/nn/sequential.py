"""
Sequential feed-forward model.

A `Sequential` holds an ordered list of `Module` layers. The forward
pass runs the input through each layer in order; the backward pass
walks the layers in reverse, threading the upstream gradient through
each layer's `backward`.

The loss is intentionally NOT part of the Sequential. To train, you call:

    logits  = model(x)             # forward through layers
    loss    = loss_fn(logits, y)   # compute scalar loss
    optimizer.zero_grad()          # clear gradient buffers
    dlogits = loss_fn.backward()   # gradient w.r.t. logits
    model.backward(dlogits)        # backprop through the model
    optimizer.step()               # one SGD step

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from collections.abc import Iterable

import numpy as np

from .module import Module


class Sequential(Module):
    """
    Sequential container of `Module` layers, mirroring
    `torch.nn.Sequential`.

    Constructor argument:
        layers -- list of Module instances applied in order during
                  forward and in reverse during backward
    """

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Runs the input through each layer in order.

        Arguments:
            x -- input batch

        Returns:
            output of the final layer
        """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Walks the layers in reverse, calling each layer's `backward`
        with the gradient flowing back from the next layer.

        Arguments:
            dout -- gradient of the loss w.r.t. the model's output

        Returns:
            gradient of the loss w.r.t. the model's input
        """
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(
        self,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """
        Yields every `(parameter, gradient)` pair from every layer
        that owns trainable parameters. Layers without parameters
        (e.g. `ReLU`) contribute nothing.
        """
        for layer in self.layers:
            yield from layer.parameters()

    def train(self) -> None:
        """Sets this container and every child layer to training mode."""
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        """Sets this container and every child layer to evaluation mode."""
        super().eval()
        for layer in self.layers:
            layer.eval()
