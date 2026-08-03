"""Build-and-train helper that integrates with the model cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from data import ArrayDataset, Dataset, Subset, random_split
from datasets.mnist import MNIST
from datasets.toy import (
    Checkerboard,
    Circles,
    Curve,
    Linear as LinearDataset,
    Moons,
    Spiral,
    XOR,
    true_curve,
)
from evaluation import (
    Accuracy,
    GradWeightRatio,
    GradientNorm,
    Metric,
    MetricCollection,
    UpdateMagnitude,
    WeightNorm,
)
from nn import (
    Adam,
    CrossEntropy,
    MLP,
    MSE,
    Module,
    SGD,
)
from nn.linear import Linear
from transforms import Compose, Normalize, ToArray
from training.model_cache import ExperimentConfig, ModelCache
from training.train_ffnn import train_ffnn

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

_MetricFactory = Callable[[Module, float], Metric]

_METRIC_REGISTRY: dict[str, _MetricFactory] = {
    "accuracy": lambda model, lr: Accuracy(),
    "grad_norm": lambda model, lr: GradientNorm(model),
    "weight_norm": lambda model, lr: WeightNorm(model),
    "grad_weight_ratio": lambda model, lr: GradWeightRatio(model),
    "update_magnitude": lambda model, lr: UpdateMagnitude(model, lr),
}

_LOSS_REGISTRY: dict[str, type[Module]] = {
    "cross_entropy": CrossEntropy,
    "mse": MSE,
}

_TRANSFORM_REGISTRY: dict[
    str, Callable[..., Callable[[np.ndarray], np.ndarray]]
] = {
    "to_array": ToArray,
    "normalize": Normalize,
}

_TOY_REGISTRY: dict[str, Callable[..., Dataset]] = {
    "moons": Moons,
    "circles": Circles,
    "xor": XOR,
    "linear": LinearDataset,
    "checkerboard": Checkerboard,
    "spiral": Spiral,
}


def _make_sgd(config: ExperimentConfig, model: MLP) -> SGD:
    return SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )


def _make_adam(config: ExperimentConfig, model: MLP) -> Adam:
    return Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


_OPTIMIZER_REGISTRY: dict[
    str, Callable[[ExperimentConfig, MLP], SGD | Adam]
] = {
    "sgd": _make_sgd,
    "adam": _make_adam,
}


def get_network(
    config: ExperimentConfig,
    *,
    cache: ModelCache | None = None,
    rebuild: bool = False,
) -> tuple[MLP, dict[str, list[float]]]:
    """
    Train a model described by `config`, or load it from `cache` if an
    entry already exists.

    Datasets are built internally from `config` via `_build_dataset`.
    `ExperimentConfig` is the single source of truth for the data, so
    the cache key fully determines the training data.

    Arguments:
        config  -- full experiment specification; config.metrics lists
                   the metric names to track during training; data fields
                   (`dataset`, `transforms`, `n_train`, `n_samples`,
                   `noise`, `data_seed`, `val_points`, `weight_scale`)
                   determine the datasets
        cache   -- optional ModelCache; pass None to skip caching
        rebuild -- if True, ignore any existing cache entry and retrain

    Returns:
        (trained_model, training_history) — history is {} when loaded
        from cache without a previously saved history.

    The three branches, all returning the same tuple type:
        1. Cache hit   → load weights from disk, return immediately.
        2. Cache miss  → train, save to cache, return.
        3. No cache    → train (nothing is saved), return.
    """
    if cache is not None and not rebuild and cache.has(config):
        return cache.load(config)

    train_ds, val_ds = _build_dataset(config)
    model = _build_model(config)
    optimizer = _build_optimizer(config, model)
    loss_fn = _build_loss(config.loss)
    metric_collection = _build_metrics(
        config.metrics or None, model, config.lr
    )

    history = train_ffnn(
        model,
        train_ds,
        val_ds,
        optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        loss_fn=loss_fn,
        seed=config.seed,
        metrics=metric_collection,
    )

    if cache is not None:
        cache.save(config, model, history)
    return model, history


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_transform(
    specs: list[dict[str, Any]],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """
    Build a transform (or chain of transforms) from a list of spec dicts.

    Each entry in `specs` is a single-key dict mapping a transform name
    to a dict of keyword arguments, e.g. ``{"to_array": {}}`` or
    ``{"normalize": {"mean": 0.1307, "std": 0.3081}}``.

    Registry:
        ``"to_array"``  → :class:`~transforms.ToArray` (**params)
        ``"normalize"`` → :class:`~transforms.Normalize` (**params)

    Returns ``None`` for an empty list, the single transform for one
    entry, or a :class:`~transforms.Compose` wrapping all entries for
    two or more.
    """
    if not specs:
        return None
    instances: list[Callable[[np.ndarray], np.ndarray]] = []
    for spec in specs:
        for name, params in spec.items():
            factory = _TRANSFORM_REGISTRY.get(name)
            if factory is None:
                raise ValueError(f"Unknown transform: {name!r}")
            instances.append(factory(**params))
    if len(instances) == 1:
        return instances[0]
    return Compose(instances)


def _build_mnist_split(
    config: ExperimentConfig,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    """Build MNIST train/val split from config."""
    if config.val_points is None:
        raise ValueError("val_points must be set for the mnist dataset")
    transform = _build_transform(config.transforms)
    full = MNIST(_DATA_ROOT, train=True, transform=transform)
    pool_size = len(full) - config.val_points
    train_pool, val_ds = random_split(
        full, [pool_size, config.val_points], seed=split_seed
    )
    if config.n_train is not None:
        train_ds: Dataset = Subset(train_pool, list(range(config.n_train)))
    else:
        train_ds = train_pool
    return train_ds, val_ds


def _build_curve_split(
    config: ExperimentConfig,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    """Build noisy curve train set and clean grid val set."""
    n = config.n_train if config.n_train is not None else config.n_samples
    noise = config.noise if config.noise is not None else 0.2
    curve_train: Dataset = Curve(n=n, noise=noise, seed=split_seed)
    val_points = config.val_points if config.val_points is not None else 300
    grid = np.linspace(0.0, 1.0, val_points).reshape(-1, 1)
    curve_val: Dataset = ArrayDataset(grid, true_curve(grid))
    return curve_train, curve_val


def _build_toy_split(
    config: ExperimentConfig,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    """Build a 2-D toy dataset train/val split from config."""
    if config.n_samples is None:
        raise ValueError(
            f"n_samples must be set for toy dataset {config.dataset!r}"
        )
    if config.val_points is None:
        raise ValueError(
            f"val_points must be set for toy dataset {config.dataset!r}"
        )
    noise = config.noise if config.noise is not None else 0.1
    cls = _TOY_REGISTRY[config.dataset]
    full_toy: Dataset = cls(n=config.n_samples, noise=noise, seed=split_seed)
    toy_train, toy_val = random_split(
        full_toy,
        [config.n_samples - config.val_points, config.val_points],
        seed=split_seed,
    )
    return toy_train, toy_val


def _build_dataset(
    config: ExperimentConfig,
) -> tuple[Dataset, Dataset]:
    """
    Build ``(train_ds, val_ds)`` from the data fields of `config`.

    Dispatches on ``config.dataset``:

    * ``"mnist"`` — loads the 60k MNIST training split, carves off
      ``val_points`` random samples as the validation set, and
      optionally sub-samples ``n_train`` examples from the remainder.
    * 2-D toys (``"moons"``, ``"circles"``, ``"xor"``, ``"linear"``,
      ``"checkerboard"``, ``"spiral"``) — generates
      ``n_samples`` points and splits ``val_points`` off at random.
    * ``"curve"`` (regression exception) — builds ``n_train`` noisy
      sine samples as the training set and a clean grid of ``val_points``
      points as the validation set; **no random holdout is taken**.

    The ``data_seed`` field (falling back to ``seed`` when ``None``)
    controls all random splits. Phase 3b callers must not pass datasets
    directly; they must set the relevant config fields instead.
    """
    split_seed = (
        config.data_seed if config.data_seed is not None else config.seed
    )
    if config.dataset == "mnist":
        return _build_mnist_split(config, split_seed)
    if config.dataset == "curve":
        return _build_curve_split(config, split_seed)
    if config.dataset in _TOY_REGISTRY:
        return _build_toy_split(config, split_seed)
    raise ValueError(f"Unknown dataset: {config.dataset!r}")


def _build_test(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST test split for callers that report test accuracy.

    Applies the same transform chain as ``_build_dataset`` so test
    features are preprocessed identically to training features.

    Arguments:
        config -- experiment config; only ``"mnist"`` is supported

    Returns:
        ``(X_test, y_test)`` as float64 / int64 numpy arrays.

    Raises ValueError for non-mnist datasets.
    """
    if config.dataset != "mnist":
        raise ValueError(
            f"_build_test is only supported for mnist, got {config.dataset!r}"
        )
    transform = _build_transform(config.transforms)
    test = MNIST(_DATA_ROOT, train=False)
    X_test = test.X.astype(np.float64)
    if transform is not None:
        X_test = np.stack([transform(test.X[i]) for i in range(len(test.X))])
    return X_test, test.y


def _build_model(config: ExperimentConfig) -> MLP:
    model = MLP(
        list(config.layer_sizes),
        config.activation,
        dropout=config.dropout,
        seed=config.seed,
    )
    if config.weight_scale != 1.0:
        for layer in model.net.layers:
            if isinstance(layer, Linear):
                layer.W *= config.weight_scale
    return model


def _build_optimizer(config: ExperimentConfig, model: MLP) -> SGD | Adam:
    factory = _OPTIMIZER_REGISTRY.get(config.optimizer)
    if factory is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer!r}")
    return factory(config, model)


def _build_loss(loss_name: str) -> Module:
    cls = _LOSS_REGISTRY.get(loss_name)
    if cls is None:
        raise ValueError(f"Unknown loss: {loss_name!r}")
    return cls()


def _build_metrics(
    metrics: list[str] | None, model: Module, lr: float
) -> MetricCollection:
    if metrics is None:
        return MetricCollection([])
    instances = []
    for name in metrics:
        factory = _METRIC_REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"Unknown metric: {name!r}")
        instances.append(factory(model, lr))
    return MetricCollection(instances)
