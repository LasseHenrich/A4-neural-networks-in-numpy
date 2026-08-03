"""Regression experiment: MLP curve fitting with MSE loss."""

from pathlib import Path

import numpy as np

from plots import plot_regression_panels
from training.model_cache import ExperimentConfig
from training.get_network import _build_dataset, get_network


def run_regression_experiment(
    data_seed: int = 0,
    n_train: int = 25,
    epochs: int = 5000,
    batch_size: int = 25,
    noise: float = 0.20,
    save_path: Path = Path("results/figures/regression_fit.png"),
    use_cache: bool = True,
) -> None:
    """
    Fits three MLP regressors to a noisy sine curve and saves a
    comparison plot.

    Configs: hidden=[1], hidden=[8, 8], hidden=[64, 64, 64]. Each
    model uses MSE loss; training error is measured on the noisy
    training points, test error on the noiseless reference curve.

    Each config uses a fixed per-config model seed chosen to give
    representative behaviour (slope for [1], good fit for [8,8],
    wiggly low-train for [64,64,64]).

    Regression bypasses the model cache — each run always retrains.

    Arguments:
        data_seed  -- random seed for data generation
        n_train    -- number of training points
        epochs     -- number of training epochs per config
        batch_size -- mini-batch size (default = n_train for full batch)
        noise      -- standard deviation of Gaussian noise added to y
        save_path  -- where to save the figure
        use_cache  -- ignored; regression always retrains (no caching)
    """
    # (hidden sizes, model_seed) — seeds chosen for representative fits
    hidden_configs: list[tuple[list[int], int]] = [
        ([1], 3),
        ([8, 8], 2),
        ([64, 64, 64], 0),
    ]
    panels: list[dict[str, object]] = []

    for hidden, model_seed in hidden_configs:
        config = ExperimentConfig(
            layer_sizes=tuple([1, *hidden, 1]),
            activation="relu",
            optimizer="adam",
            lr=0.01,
            epochs=epochs,
            batch_size=batch_size,
            loss="mse",
            seed=model_seed,
            dataset="curve",
            n_train=n_train,
            noise=noise,
            data_seed=data_seed,
            val_points=300,
        )
        model, _ = get_network(config, cache=None)

        train_ds, val_ds = _build_dataset(config)
        X_train = train_ds.X
        y_train = train_ds.y
        x_grid = val_ds.X
        y_true = val_ds.y

        train_mse = float(np.mean((model(X_train) - y_train) ** 2))
        test_mse = float(np.mean((model(x_grid) - y_true) ** 2))
        y_pred = model(x_grid)
        panels.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "x_grid": x_grid,
                "y_true": y_true,
                "y_pred": y_pred,
                "title": (
                    f"hidden={hidden}, "
                    f"train={train_mse:.4f}, "
                    f"test={test_mse:.4f}"
                ),
            }
        )

    plot_regression_panels(panels, save_path)
    print(f"Saved {save_path}")
