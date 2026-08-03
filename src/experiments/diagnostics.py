"""Gradient and weight diagnostic experiment on toy data and MNIST."""

from pathlib import Path

from plots.diagnostics import plot_diagnostic_trajectories
from training.get_network import get_network
from training.model_cache import ExperimentConfig

_DIAGNOSTIC_METRICS: list[str] = [
    "accuracy",
    "grad_norm",
    "weight_norm",
    "grad_weight_ratio",
    "update_magnitude",
]

# (condition_name, base_learning_rate, use_deep_bad_init)
_CONDITIONS: list[tuple[str, float, bool]] = [
    ("healthy", 0.05, False),
    ("lr_too_high", 10.0, False),
    ("lr_too_low", 1e-5, False),
    ("deep_bad_init", 0.05, True),
]

_CONDITION_NAMES: list[str] = [c[0] for c in _CONDITIONS]

# Per-dataset learning-rate overrides for lr_too_high.
# Toy moons (2-D, unit-scale inputs): lr=10 is moderately too high —
# the net survives but loss oscillates and weight norm explodes.
# MNIST subset (784-D, [0,1]-scaled inputs): the effective step is
# much larger, so lr=1.5 already produces sustained instability without
# collapsing gradients entirely.
_LR_TOO_HIGH_OVERRIDES: dict[str, float] = {
    "toy (moons)": 10.0,
    "MNIST subset": 1.5,
}

# Weight-scale applied to every Linear W in the deep_bad_init condition.
# 0.1 (rather than 0.01) keeps the network alive enough to show gradual
# improvement while still starting ~5x below the healthy weight norm.
_DEEP_BAD_INIT_SCALE: float = 0.1


def _build_condition_config(
    condition: str,
    dataset_name: str,
    dataset_key: str,
    base_sizes: list[int],
    data_fields: dict[str, object],
    base_lr: float,
    deep_bad_init: bool,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> ExperimentConfig:
    """
    Build an `ExperimentConfig` for one (dataset, condition) cell.

    Resolves the learning rate (using per-dataset overrides for the
    ``lr_too_high`` condition) and the layer sizes / weight scale
    (expanding to a four-hidden-layer network for ``deep_bad_init``).
    All other training hyperparameters are forwarded unchanged.
    """
    if condition == "lr_too_high":
        lr = _LR_TOO_HIGH_OVERRIDES.get(dataset_name, base_lr)
    else:
        lr = base_lr

    if deep_bad_init:
        n_in = base_sizes[0]
        n_out = base_sizes[-1]
        layer_sizes: tuple[int, ...] = (n_in, 64, 64, 64, 64, n_out)
        weight_scale = _DEEP_BAD_INIT_SCALE
    else:
        layer_sizes = tuple(base_sizes)
        weight_scale = 1.0

    return ExperimentConfig(
        layer_sizes=layer_sizes,
        activation="relu",
        optimizer="sgd",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        loss="cross_entropy",
        seed=seed,
        dataset=dataset_key,
        weight_scale=weight_scale,
        metrics=_DIAGNOSTIC_METRICS,
        **data_fields,  # type: ignore[arg-type]
    )


def run_diagnostics_experiment(
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 64,
    save_path: Path = Path("results/figures/training_diagnostics.png"),
) -> None:
    """
    Trains four contrasting configurations on a 2-D toy dataset and a
    small MNIST subset. Collects per-epoch gradient and weight
    diagnostics for each configuration and saves a trajectory plot.

    Conditions:
        healthy      -- lr=0.05, single hidden layer, default init
        lr_too_high  -- lr per dataset (toy: 10.0, MNIST: 1.5);
                        single hidden layer, default init
        lr_too_low   -- lr=1e-5, single hidden layer, default init
        deep_bad_init -- lr=0.05, four hidden layers,
                         W scaled by _DEEP_BAD_INIT_SCALE (0.1)

    Arguments:
        seed       -- random seed for data, model init, and training
        epochs     -- number of training epochs per condition
        batch_size -- mini-batch size
        save_path  -- where to write the output figure
    """
    # (dataset_name, dataset key, base layer_sizes, data config fields)
    datasets: list[tuple[str, str, list[int], dict[str, object]]] = [
        (
            "toy (moons)",
            "moons",
            [2, 64, 2],
            {
                "n_samples": 300,
                "noise": 0.1,
                "val_points": 60,
                "data_seed": seed,
                "transforms": [],
            },
        ),
        (
            "MNIST subset",
            "mnist",
            [784, 64, 10],
            {
                "n_train": 2000,
                "val_points": 500,
                "data_seed": seed,
                "transforms": [{"to_array": {}}],
            },
        ),
    ]

    results: dict[str, dict[str, dict[str, list[float]]]] = {}

    for dataset_name, dataset_key, base_sizes, data_fields in datasets:
        results[dataset_name] = {}

        for condition, base_lr, deep_bad_init in _CONDITIONS:
            config = _build_condition_config(
                condition,
                dataset_name,
                dataset_key,
                base_sizes,
                data_fields,
                base_lr,
                deep_bad_init,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
            )
            _, history = get_network(config, cache=None)
            results[dataset_name][condition] = history

    plot_diagnostic_trajectories(
        results=results,
        condition_names=_CONDITION_NAMES,
        save_path=save_path,
    )
    print(f"Saved {save_path}")
