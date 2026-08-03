"""
Subset and random_split utilities.

Mirrors torch.utils.data.Subset and torch.utils.data.random_split.
"""

import numpy as np

from .dataset import Dataset


class Subset(Dataset):
    """
    A Dataset backed by a subset of indices into another Dataset.

    Mirrors torch.utils.data.Subset.

    Constructor arguments:
        dataset -- parent Dataset
        indices -- integer indices into `dataset`
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]


def random_split(
    dataset: Dataset,
    lengths: list[int],
    seed: int = 0,
) -> list[Subset]:
    """
    Randomly splits a Dataset into non-overlapping Subsets.

    Mirrors torch.utils.data.random_split.

    Arguments:
        dataset -- Dataset to partition
        lengths -- size of each split; must sum to len(dataset)
        seed    -- random seed controlling the index permutation

    Returns:
        list[Subset], one per entry in lengths

    Raises ValueError if sum(lengths) != len(dataset) or any
    length is not positive.
    """
    n = len(dataset)
    if sum(lengths) != n:
        raise ValueError(
            f"sum(lengths) must equal len(dataset); "
            f"got sum={sum(lengths)}, len={n}"
        )
    if any(length <= 0 for length in lengths):
        raise ValueError("all lengths must be positive integers")

    rng = np.random.default_rng(seed)
    perm: list[int] = rng.permutation(n).tolist()

    subsets: list[Subset] = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, perm[offset : offset + length]))
        offset += length
    return subsets
