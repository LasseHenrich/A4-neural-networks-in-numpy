"""
Feature transforms applied to dataset samples.

Each transform is a callable that maps a NumPy array to a NumPy array,
so transforms can be chained with `Compose` and handed to a dataset.
The naming follows torchvision:

    Normalize -- standardize to zero mean / unit std: (x - mean) / std
    ToArray   -- min-max rescale a fixed range to [0, 1]
    Compose   -- apply a list of transforms left to right

Mirrors `torchvision.transforms`.
"""

from collections.abc import Callable, Sequence

import numpy as np


class Compose:
    """
    Chains several transforms into one.

    Calling the composed transform applies each member in order, feeding
    the output of one into the next.

    Constructor arguments:
        transforms -- the transforms to apply, in order
    """

    def __init__(
        self,
        transforms: Sequence[Callable[[np.ndarray], np.ndarray]],
    ) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x


class Normalize:
    """
    Standardizes an array with `(x - mean) / std`.

    This is torchvision's `Normalize`: it shifts by `mean` and scales by
    `std`, giving (for matching statistics) zero mean and unit variance.
    The defaults are the MNIST training-set mean and std computed on
    pixels already scaled to [0, 1].

    Constructor arguments:
        mean -- value subtracted from the input
        std  -- value the shifted input is divided by; must be non-zero

    Raises ValueError if `std` is zero.
    """

    def __init__(self, mean: float = 0.1307, std: float = 0.3081) -> None:
        if std == 0:
            raise ValueError("std must be non-zero")
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class ToArray:
    """
    Maps pixels from [0, 255] to [0, 1] by dividing by 255.

    This is min-max rescaling over a fixed range: `(x - low) /
    (high - low)`. The name mirrors torchvision's `ToTensor` (which
    also scales to [0, 1]) but stays accurate for a numpy-only repo.

    Constructor arguments:
        low  -- value mapped to 0
        high -- value mapped to 1; must be greater than `low`

    Raises ValueError if `high` is not greater than `low`.
    """

    def __init__(self, low: float = 0.0, high: float = 255.0) -> None:
        if high <= low:
            raise ValueError("high must be greater than low")
        self.low = low
        self.high = high

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.low) / (self.high - self.low)
