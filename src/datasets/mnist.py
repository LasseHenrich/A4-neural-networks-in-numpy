"""
MNIST loading and dataset.

Downloads the MNIST handwritten-digit dataset via
`sklearn.datasets.fetch_openml` and caches it as a `.npz` file.

    MNIST -- a `Dataset` (torchvision style) exposing the train or
             test split with raw [0, 255] pixels and an optional
             transform

This file is provided. You should not need to edit it.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from data.dataset import Dataset

_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _ROOT / "data" / "raw" / "mnist.npz"


def _download_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Fetches MNIST from OpenML and returns (X, y) as numpy arrays."""
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
    )
    X = np.asarray(bundle.data, dtype=np.float32)
    y = np.asarray(bundle.target, dtype=np.int64)
    return X, y


def _load_or_download(cache: Path = _CACHE) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y), pulling from `cache` if it exists."""
    if cache.exists():
        with np.load(cache) as data:
            return data["X"], data["y"]
    X, y = _download_mnist()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


class MNIST(Dataset):
    """
    The MNIST handwritten-digit dataset as a `Dataset`.

    Loads MNIST from a cache under `root`, downloading it there on first
    use, and exposes either the training or the test split. Each item is
    an `(image, label)` pair, where `image` holds raw [0, 255] float32
    pixels of shape (784,). Pass a transform — for example
    `Compose([ToArray(), Normalize()])` — to rescale the pixels.

    Mirrors `torchvision.datasets.MNIST`.

    Constructor arguments:
        root      -- directory holding (or to receive) the MNIST cache
        train     -- if True, expose the 60000-image training split;
                     otherwise the 10000-image test split
        transform -- optional callable applied to each image
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        X, y = _load_or_download(Path(root) / "mnist.npz")
        if train:
            self.X = X[:60000]
            self.y = y[:60000]
        else:
            self.X = X[60000:]
            self.y = y[60000:]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]
