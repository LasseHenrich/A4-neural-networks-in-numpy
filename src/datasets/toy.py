"""Toy dataset generators for 2-D classification and 1-D regression experiments."""

import numpy as np

from data.dataset import ArrayDataset


def make_moons(
    n: int = 200,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two interlocking half-moon point clouds.

    Arguments:
        n     -- total number of points
        noise -- standard deviation of Gaussian noise added to X
        seed  -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    theta0 = np.linspace(0, np.pi, n0)
    theta1 = np.linspace(0, np.pi, n1)

    X0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    X1 = np.column_stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5])

    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate(
        [np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)]
    )
    return X.astype(np.float64), y


def make_circles(
    n: int = 200,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two concentric circles.

    Arguments:
        n      -- total number of points
        noise  -- standard deviation of Gaussian noise added to X
        factor -- radius of the inner circle (outer radius is 1)
        seed   -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    theta0 = np.linspace(0, 2 * np.pi, n0)
    theta1 = np.linspace(0, 2 * np.pi, n1)

    X0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    X1 = np.column_stack([factor * np.cos(theta1), factor * np.sin(theta1)])

    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate(
        [np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)]
    )
    return X.astype(np.float64), y


def make_xor(
    n: int = 200,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a four-quadrant XOR pattern.

    Class 0 occupies Q1 and Q3 (same-sign coordinates); class 1
    occupies Q2 and Q4.

    Arguments:
        n     -- total number of points
        noise -- standard deviation of Gaussian noise added to X
        seed  -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 2))
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(np.int64)
    X = X + rng.normal(0, noise, X.shape)
    return X.astype(np.float64), y


def make_linear(
    n: int = 200,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns linearly separable data (two Gaussians split by a diagonal).

    Arguments:
        n     -- total number of points
        noise -- standard deviation of Gaussian noise added to X
        seed  -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    X = X + rng.normal(0, noise, X.shape)
    return X.astype(np.float64), y


def make_checkerboard(
    n: int = 400,
    grid: int = 4,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a checkerboard pattern of alternating-class cells.

    Points are sampled uniformly from [-1, 1]^2 and assigned to class 0
    or 1 based on the parity of their grid cell, producing `grid * grid`
    alternating cells.

    Arguments:
        n     -- total number of points
        grid  -- number of cells along each axis (total cells = grid^2)
        noise -- standard deviation of Gaussian noise added to X
        seed  -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 2))
    cells = np.floor(grid * (X + 1.0) / 2.0).astype(np.int64)
    y = ((cells[:, 0] + cells[:, 1]) % 2).astype(np.int64)
    X = X + rng.normal(0, noise, X.shape)
    return X.astype(np.float64), y


def make_spiral(
    n: int = 300,
    noise: float = 0.2,
    turns: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two interleaved spiral arms.

    Each class traces a spiral arm from the origin outward; the two
    arms are offset by pi radians so they wind around each other. The
    smoothly curved boundary cannot be traced by a single straight
    line and requires multiple ReLU folds to follow tightly.

    Arguments:
        n     -- total number of points
        noise -- standard deviation of Gaussian noise added to X
        turns -- number of turns each arm makes from center to edge
        seed  -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    t0 = np.linspace(0.0, 1.0, n0)
    t1 = np.linspace(0.0, 1.0, n1)
    theta0 = turns * 2 * np.pi * t0
    theta1 = turns * 2 * np.pi * t1 + np.pi

    X0 = np.column_stack([t0 * np.cos(theta0), t0 * np.sin(theta0)])
    X1 = np.column_stack([t1 * np.cos(theta1), t1 * np.sin(theta1)])

    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate(
        [np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)]
    )
    return X.astype(np.float64), y


def make_blobs(
    n: int = 300,
    centers: int = 3,
    std: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gaussian blobs for multiclass classification.

    Cluster centers are placed at equal angles on a circle of radius 3.
    Points are assigned to clusters round-robin and then sampled from a
    Gaussian of standard deviation `std` around each center.

    Arguments:
        n       -- total number of points
        centers -- number of clusters / classes
        std     -- standard deviation of each Gaussian cluster
        seed    -- random seed

    Returns:
        X of shape (n, 2) float64, y of shape (n,) int64
    """
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(centers) / centers
    centers_xy = np.column_stack([3 * np.cos(angles), 3 * np.sin(angles)])
    y = (np.arange(n) % centers).astype(np.int64)
    X = centers_xy[y] + rng.normal(0, std, (n, 2))
    return X.astype(np.float64), y


def make_curve(
    n: int = 25,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a noisy sine curve for regression.

    Arguments:
        n     -- number of points
        noise -- standard deviation of Gaussian noise added to y
        seed  -- random seed

    Returns:
        X of shape (n, 1) float64, y of shape (n, 1) float64
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, 1)).astype(np.float64)
    y = (np.sin(2 * np.pi * X) + rng.normal(0, noise, X.shape)).astype(
        np.float64
    )
    return X, y


def true_curve(x_grid: np.ndarray) -> np.ndarray:
    """
    Returns the noiseless sine reference line at `x_grid` points.

    Arguments:
        x_grid -- input points, any shape

    Returns:
        np.sin(2 * pi * x_grid), same shape as x_grid
    """
    return np.sin(2 * np.pi * x_grid)


# ---------------------------------------------------------------------------
# Dataset classes (torchvision-style wrappers around the generators above)
# ---------------------------------------------------------------------------


class Moons(ArrayDataset):
    """Two interlocking half-moon point clouds."""

    def __init__(
        self,
        n: int = 200,
        noise: float = 0.1,
        seed: int = 0,
    ) -> None:
        X, y = make_moons(n=n, noise=noise, seed=seed)
        super().__init__(X, y)


class Circles(ArrayDataset):
    """Two concentric circles."""

    def __init__(
        self,
        n: int = 200,
        noise: float = 0.1,
        factor: float = 0.5,
        seed: int = 0,
    ) -> None:
        X, y = make_circles(n=n, noise=noise, factor=factor, seed=seed)
        super().__init__(X, y)


class XOR(ArrayDataset):
    """Four-quadrant XOR pattern."""

    def __init__(
        self,
        n: int = 200,
        noise: float = 0.1,
        seed: int = 0,
    ) -> None:
        X, y = make_xor(n=n, noise=noise, seed=seed)
        super().__init__(X, y)


class Linear(ArrayDataset):
    """Linearly separable data (two Gaussians split by a diagonal)."""

    def __init__(
        self,
        n: int = 200,
        noise: float = 0.1,
        seed: int = 0,
    ) -> None:
        X, y = make_linear(n=n, noise=noise, seed=seed)
        super().__init__(X, y)


class Checkerboard(ArrayDataset):
    """Checkerboard pattern of alternating-class cells."""

    def __init__(
        self,
        n: int = 400,
        grid: int = 4,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        X, y = make_checkerboard(n=n, grid=grid, noise=noise, seed=seed)
        super().__init__(X, y)


class Spiral(ArrayDataset):
    """Two interleaved spiral arms."""

    def __init__(
        self,
        n: int = 300,
        noise: float = 0.2,
        turns: float = 1.5,
        seed: int = 0,
    ) -> None:
        X, y = make_spiral(n=n, noise=noise, turns=turns, seed=seed)
        super().__init__(X, y)


class Blobs(ArrayDataset):
    """Gaussian blobs for multiclass classification."""

    def __init__(
        self,
        n: int = 300,
        centers: int = 3,
        std: float = 1.0,
        seed: int = 0,
    ) -> None:
        X, y = make_blobs(n=n, centers=centers, std=std, seed=seed)
        super().__init__(X, y)


class Curve(ArrayDataset):
    """Noisy sine curve for regression."""

    def __init__(
        self,
        n: int = 25,
        noise: float = 0.2,
        seed: int = 0,
    ) -> None:
        X, y = make_curve(n=n, noise=noise, seed=seed)
        super().__init__(X, y)
