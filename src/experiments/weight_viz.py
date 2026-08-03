"""First-layer weight visualization experiment on MNIST."""

from pathlib import Path

from evaluation import accuracy
from plots import plot_weight_blocks
from training.get_network import _build_test, get_network
from training.model_cache import ExperimentConfig, ModelCache

_NORM_TRANSFORMS: list[dict[str, dict]] = [
    {"to_array": {}},
    {"normalize": {}},
]


def run_weight_viz_experiment(
    seed: int = 0,
    hidden_sizes: tuple[int, ...] = (4, 16, 64),
    epochs: int = 40,
    batch_size: int = 128,
    val_size: int = 5000,
    save_path_baseline: Path = Path(
        "results/figures/learned_features_baseline.png"
    ),
    save_path_regularized: Path = Path(
        "results/figures/learned_features_regularized.png"
    ),
    use_cache: bool = True,
) -> None:
    """
    Trains three networks with different first-layer widths on MNIST
    and visualises the learned input weights as filter images.

    Produces two figures side-by-side for comparison:
      - baseline     (weight_decay=1e-4): typical training regularization
      - regularized  (weight_decay=1e-2): strong L2 for visual clarity

    For each hidden size H, a 784→H→10 network is trained. The first
    Linear layer's weight matrix (shape 784 × H) is extracted and
    rendered as a grid of 28×28 filter images.

    Training uses a held-out validation split (full 60k minus val_size)
    rather than the raw test set.

    Arguments:
        seed                 -- random seed for data splits, model init,
                                and training shuffles
        hidden_sizes         -- first hidden-layer widths to sweep over
        epochs               -- number of training epochs per run
        batch_size           -- mini-batch size
        val_size             -- number of validation examples carved from
                                the training split
        save_path_baseline   -- output path for the wd=1e-4 figure
        save_path_regularized -- output path for the wd=1e-2 figure
        use_cache            -- load cached model weights when available;
                                set False to force retraining
    """
    cache = ModelCache() if use_cache else None

    baseline_blocks: list[dict[str, object]] = []
    regularized_blocks: list[dict[str, object]] = []

    for H in hidden_sizes:
        for weight_decay, blocks in (
            (1e-4, baseline_blocks),
            (1e-2, regularized_blocks),
        ):
            config = ExperimentConfig(
                layer_sizes=(784, H, 10),
                activation="relu",
                optimizer="sgd",
                lr=0.1,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                loss="cross_entropy",
                seed=seed,
                dataset="mnist",
                val_points=val_size,
                transforms=_NORM_TRANSFORMS,
            )
            model, _ = get_network(config, cache=cache)
            X_test, y_test = _build_test(config)
            test_acc = float(accuracy(model(X_test).argmax(axis=1), y_test))
            W = model.net.layers[0].W
            blocks.append({"W": W, "H": H, "test_acc": test_acc})

    plot_weight_blocks(baseline_blocks, save_path_baseline)
    print(f"Saved {save_path_baseline}")
    plot_weight_blocks(regularized_blocks, save_path_regularized)
    print(f"Saved {save_path_regularized}")


def run_reg_sweep(
    H: int,
    weight_decays: tuple[float, ...] = (1e-4,),
    dropouts: tuple[float, ...] = (0.0,),
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 128,
    val_size: int = 5000,
    save_path: Path = Path("results/figures/reg_sweep.png"),
    use_cache: bool = True,
) -> None:
    """
    Sweep regularization settings for a fixed first-layer width H.

    Trains one 784→H→10 network per (weight_decay, dropout) pair and
    visualises the first-layer weight filters side-by-side so the effect
    of each regularization setting can be compared directly.

    Training uses a held-out validation split (full 60k minus val_size)
    rather than the raw test set.

    Arguments:
        H             -- first hidden-layer width (fixed across the sweep)
        weight_decays -- L2 weight-decay values to sweep
        dropouts      -- dropout probabilities to sweep (applied after the
                         hidden activation, before the output layer)
        seed          -- random seed for splits, init, and training
        epochs        -- training epochs per run
        batch_size    -- mini-batch size
        val_size      -- number of validation examples
        save_path     -- output path for the comparison figure
        use_cache     -- load cached weights when available
    """
    from itertools import product as iproduct

    cache = ModelCache() if use_cache else None

    blocks: list[dict[str, object]] = []
    for wd, p in iproduct(weight_decays, dropouts):
        config = ExperimentConfig(
            layer_sizes=(784, H, 10),
            activation="relu",
            optimizer="sgd",
            lr=0.1,
            weight_decay=wd,
            dropout=p,
            epochs=epochs,
            batch_size=batch_size,
            loss="cross_entropy",
            seed=seed,
            dataset="mnist",
            val_points=val_size,
            transforms=_NORM_TRANSFORMS,
        )
        model, _ = get_network(config, cache=cache)
        X_test, y_test = _build_test(config)
        test_acc = float(accuracy(model(X_test).argmax(axis=1), y_test))
        W = model.net.layers[0].W
        title = f"wd={wd:.0e}, p={p:.1f}\nacc={test_acc:.3f}"
        blocks.append({"W": W, "H": H, "test_acc": test_acc, "title": title})

    plot_weight_blocks(blocks, save_path)
    print(f"Saved {save_path}")
