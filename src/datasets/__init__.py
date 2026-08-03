"""Concrete datasets."""

from .mnist import MNIST
from .toy import (
    Blobs,
    Checkerboard,
    Circles,
    Curve,
    Linear,
    Moons,
    Spiral,
    XOR,
)

__all__ = [
    "MNIST",
    "Blobs",
    "Checkerboard",
    "Circles",
    "Curve",
    "Linear",
    "Moons",
    "Spiral",
    "XOR",
]
