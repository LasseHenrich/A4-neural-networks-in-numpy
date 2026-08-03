"""Optimizer and regularization comparison experiment on MNIST."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from plots import plot_curve_panels
from training.get_network import get_network
from training.model_cache import ExperimentConfig, ModelCache


def _collect_runs(
    variants: dict[str, dict[str, Any]],
    base: ExperimentConfig,
    cache: ModelCache | None,
) -> dict[str, dict[str, list[float]]]:
    """
    Train (or load) one model per variant and return a runs dict.

    For each ``(label, override)`` pair in *variants*, builds a config
    via ``dataclasses.replace(base, **override)``, fetches the trained
    model and history via ``get_network``, then collects
    ``{"val": val_acc}`` and, when present, ``{"train": train_acc}``.

    Arguments:
        variants -- ordered map of label → field overrides for replace()
        base     -- shared base ExperimentConfig
        cache    -- ModelCache instance, or None to disable caching
    """
    runs: dict[str, dict[str, list[float]]] = {}
    for label, override in variants.items():
        config = replace(base, **override)
        _, history = get_network(config, cache=cache)
        run_curves: dict[str, list[float]] = {"val": history["val_acc"]}
        if "train_acc" in history:
            run_curves["train"] = history["train_acc"]
        runs[label] = run_curves
    return runs


def run_hyperparameter_experiment(
    seed: int = 0,
    n_train: int = 3000,
    n_train_reg: int = 1000,
    epochs: int = 30,
    epochs_reg: int = 60,
    batch_size: int = 128,
    val_size: int = 5000,
    save_path: Path = Path("results/figures/hyperparameters.png"),
    use_cache: bool = True,
) -> None:
    """
    Trains a 784→256→128→10 network under three optimizers (Panel A)
    and three regularization settings (Panel B), saving a two-panel
    figure.

    Panel A uses n_train examples so the optimizer curves reach high
    accuracy.  Panel B uses the smaller n_train_reg subset to induce
    strong enough overfitting that dropout and weight decay visibly
    close the train/val gap.

    Panel A — Optimizers: SGD, SGD with momentum, Adam.
    Panel B — Regularization: none, Dropout(0.3), weight-decay=2e-2.

    Arguments:
        seed        -- random seed for data splits, model init, and
                       training shuffles
        n_train     -- training examples for Panel A (optimizer curves)
        n_train_reg -- training examples for Panel B (regularization
                       curves); smaller value induces more overfitting
        epochs      -- number of training epochs per run
        batch_size  -- mini-batch size
        val_size    -- number of validation examples
        save_path   -- where to save the figure
        use_cache   -- load cached model weights when available; set
                       False to force retraining
    """
    cache = ModelCache() if use_cache else None
    norm_transforms: list[dict[str, dict]] = [
        {"to_array": {}},
        {"normalize": {}},
    ]

    base = ExperimentConfig(
        layer_sizes=(784, 256, 128, 10),
        activation="relu",
        optimizer="sgd",
        lr=0.1,
        batch_size=batch_size,
        loss="cross_entropy",
        seed=seed,
        dataset="mnist",
        val_points=val_size,
        transforms=norm_transforms,
        metrics=["accuracy"],
    )

    # --- Panel A: optimizers ---
    # Adam default lr is 1e-3; SGD variants use lr=0.1.
    panel_a_variants: dict[str, dict[str, Any]] = {
        "sgd": {},
        "momentum": {"momentum": 0.9},
        "adam": {"optimizer": "adam", "lr": 1e-3},
    }
    optimizer_runs = _collect_runs(
        panel_a_variants,
        replace(base, n_train=n_train, epochs=epochs),
        cache,
    )

    # --- Panel B: regularization ---
    panel_b_variants: dict[str, dict[str, Any]] = {
        "none": {},
        "dropout": {"dropout": 0.3},
        "weight_decay": {"weight_decay": 2e-2},
    }
    reg_runs = _collect_runs(
        panel_b_variants,
        replace(base, n_train=n_train_reg, epochs=epochs_reg),
        cache,
    )

    # --- Build figure ---
    panels = [
        {
            "title": "Optimizers",
            "runs": optimizer_runs,
            "ylabel": "Accuracy",
        },
        {
            "title": "Regularization",
            "runs": reg_runs,
            "ylabel": "Accuracy",
        },
    ]
    plot_curve_panels(panels=panels, save_path=Path(save_path))
    print(f"Saved {save_path}")
