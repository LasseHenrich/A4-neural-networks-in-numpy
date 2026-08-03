"""Input-scaling experiment on MNIST."""

from pathlib import Path

from plots import plot_scaling_lr_grid
from training.get_network import get_network
from training.model_cache import ExperimentConfig

_LAYER_SIZES = [784, 128, 10]


def run_scaling_experiment(
    seed: int = 0,
    lrs: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    epochs: int = 10,
    batch_size: int = 128,
    val_size: int = 5000,
    save_path: Path = Path("results/figures/input_scaling.png"),
) -> None:
    """
    Trains a 784→128→10 network under three input-scaling pipelines
    and across multiple learning rates, saving a grid of diagnostic
    curves.

    The figure has one column per scaling pipeline and three rows:
    validation accuracy, training gradient norm, and training update
    magnitude.  The update-magnitude plot highlights the healthy band
    [1e-3, 1e-2] to show which (pipeline, lr) combinations stay in
    a safe range.

    Pipelines:
        raw        -- raw pixels [0, 255]
        toarray    -- ToArray() applied → [0, 1]
        normalized -- Compose([ToArray(), Normalize()]) → standardized

    Arguments:
        seed       -- random seed for data splits, model init, and
                      training shuffles
        lrs        -- learning rates to sweep over
        epochs     -- number of training epochs per run
        batch_size -- mini-batch size
        val_size   -- number of validation examples carved out of the
                      training split
        save_path  -- where to save the figure
    """
    pipeline_transforms: dict[str, list[dict[str, dict]]] = {
        "raw": [],
        "toarray": [{"to_array": {}}],
        "normalized": [{"to_array": {}}, {"normalize": {}}],
    }

    # results[pipeline][lr] = {
    #   "val_acc":          list[float],
    #   "grad_norm":        list[float],
    #   "update_magnitude": list[float],
    # }
    results: dict[str, dict[float, dict[str, list[float]]]] = {}

    for name, transforms in pipeline_transforms.items():
        results[name] = {}
        for i, lr in enumerate(lrs):
            # Fresh model per (pipeline, lr) so runs are independent.
            config = ExperimentConfig(
                layer_sizes=tuple(_LAYER_SIZES),
                activation="relu",
                optimizer="sgd",
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                loss="cross_entropy",
                seed=seed + i,
                dataset="mnist",
                n_train=10000,
                val_points=val_size,
                transforms=transforms,
                metrics=["accuracy", "grad_norm", "update_magnitude"],
            )
            _, history = get_network(config)

            results[name][lr] = {
                "val_acc": history.get("val_acc", []),
                "grad_norm": history.get("train_grad_norm", []),
                "update_magnitude": history.get("train_update_magnitude", []),
            }

    plot_scaling_lr_grid(results, save_path)
    print(f"Saved {save_path}")
