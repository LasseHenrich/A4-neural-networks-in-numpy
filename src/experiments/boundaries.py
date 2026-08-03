"""Decision-boundary experiment on 2-D toy datasets."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from data import Dataset
from nn import Module
from plots import plot_boundary_panels
from training.get_network import _build_dataset, get_network  # noqa: PLC2701
from training.model_cache import ExperimentConfig, ModelCache


def _stack(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays by stacking every item in `ds`."""
    xs = [ds[i][0] for i in range(len(ds))]
    ys = [ds[i][1] for i in range(len(ds))]
    return np.array(xs), np.array(ys)


def _make_predict_fn(
    model: Module,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict function that argmax-es model output."""

    def predict_fn(grid: np.ndarray) -> np.ndarray:
        return model(grid).argmax(axis=1)

    return predict_fn


def _collect_points(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack train + val points from `config` into (X, y) arrays."""
    train_ds, val_ds = _build_dataset(config)
    Xtr, ytr = _stack(train_ds)
    Xv, yv = _stack(val_ds)
    return np.vstack([Xtr, Xv]), np.concatenate([ytr, yv])


# Per-column dataset parameters: (n_samples, noise, val_points)
_DATASET_PARAMS: dict[str, tuple[int, float, int]] = {
    "linear": (300, 0.1, 60),
    "moons": (300, 0.1, 60),
    "circles": (300, 0.1, 60),
    "xor": (300, 0.1, 60),
    "checkerboard": (400, 0.05, 80),
    "spiral": (300, 0.05, 60),
}


def run_boundary_experiment(
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 64,
    save_path: Path = Path("results/figures/decision_boundaries.png"),
    use_cache: bool = True,
) -> None:
    """
    Trains five model configurations on six toy 2-D datasets and
    saves a grid of decision-boundary plots.

    Rows correspond to model configs, ordered to make two
    pedagogical comparisons readable side-by-side:
        rows 1 & 2: linear classifier vs. two-Linear stack with no
                    activation (identical boundaries — depth without
                    non-linearity collapses)
        rows 3–5: capacity progression ([4], [16], [64,64]) showing
                  that more total ReLU folds produce more intricate
                  piecewise-linear boundaries
    Columns are the toy datasets (linear, moons, circles, xor,
    checkerboard, spiral).

    Arguments:
        seed       -- random seed for data generation, model init, and
                      training
        epochs     -- number of training epochs per run
        batch_size -- mini-batch size
        save_path  -- where to save the figure
        use_cache  -- load cached model weights when available; set
                      False to force retraining
    """
    col_labels = list(_DATASET_PARAMS.keys())
    row_labels = [
        "No hidden (1 Linear)",
        "No activation (Linear→Linear)",
        "[4] + ReLU",
        "[16] + ReLU",
        "[64, 64] + ReLU",
    ]

    # (layer_sizes, activation) per row. Row 2 uses "identity" to
    # produce a no-activation stack equivalent to Sequential([L, L]).
    row_configs: list[tuple[tuple[int, ...], str]] = [
        ((2, 2), "relu"),
        ((2, 64, 2), "identity"),
        ((2, 4, 2), "relu"),
        ((2, 16, 2), "relu"),
        ((2, 64, 64, 2), "relu"),
    ]

    cache = ModelCache() if use_cache else None
    panels: list[list[dict[str, object]]] = []

    for (layer_sizes, activation), row_label in zip(row_configs, row_labels):
        row: list[dict[str, object]] = []
        for col_label in col_labels:
            n_samples, noise, val_points = _DATASET_PARAMS[col_label]

            config = ExperimentConfig(
                layer_sizes=layer_sizes,
                activation=activation,
                optimizer="adam",
                lr=0.01,
                epochs=epochs,
                batch_size=batch_size,
                loss="cross_entropy",
                seed=seed,
                dataset=col_label,
                n_samples=n_samples,
                noise=noise,
                data_seed=seed,
                val_points=val_points,
                metrics=["accuracy"],
            )
            m, history = get_network(config, cache=cache)
            acc = history["val_acc"][-1]
            m.eval()

            # Stack all points (train + val) so the full cloud shows.
            X, y = _collect_points(config)

            row.append(
                {
                    "predict_fn": _make_predict_fn(m),
                    "X": X,
                    "y": y,
                    "title": f"acc={acc:.2f}",
                }
            )
        panels.append(row)

    plot_boundary_panels(
        panels=panels,
        row_labels=row_labels,
        col_labels=col_labels,
        save_path=save_path,
    )
    print(f"Saved {save_path}")
