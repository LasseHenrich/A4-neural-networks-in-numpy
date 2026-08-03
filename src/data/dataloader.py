"""
DataLoader: wraps a Dataset and yields mini-batches.

Mirrors the interface of `torch.utils.data.DataLoader`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import math
from collections.abc import Iterator

import numpy as np

from .dataset import ArrayDataset, Dataset


class DataLoader:
    """
    Iterates over a `Dataset` in mini-batches.

    Constructor arguments:
        dataset    -- any Dataset instance
        batch_size -- number of samples per batch; must be >= 1
        shuffle    -- if True, reshuffle indices at the start of each
                      iteration
        seed       -- seed for the RNG used when shuffling

    Raises ValueError if batch_size < 1.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng: np.random.Generator | None = (
            np.random.default_rng(seed) if shuffle else None
        )

    def __len__(self) -> int:
        """Returns the number of batches."""
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (X_batch, y_batch) pairs, one mini-batch at a time."""
        indices = np.arange(len(self.dataset))
        if self.shuffle and self.rng is not None:
            self.rng.shuffle(indices)
            
        for start_i in range(0, len(self.dataset), self.batch_size):
            batch_idx = indices[start_i : start_i + self.batch_size]
            batch = [self.dataset[i] for i in batch_idx]
            xs, ys = zip(*batch)
            X_batch = np.stack(xs)
            y_batch = np.array(ys)
            yield X_batch, y_batch
