"""Learning-rate sweep experiment on MNIST."""

from pathlib import Path

from evaluation import accuracy
from plots import plot_lr_experiment
from training.get_network import _build_test, get_network
from training.model_cache import ExperimentConfig, ModelCache

_ROOT = Path(__file__).resolve().parent.parent.parent


def run_lr_experiment(
    learning_rates: list[float] | None = None,
    hidden: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    val_size: int = 10000,
    seed: int = 0,
    save_path: Path = Path("results/figures/lr_experiment.png"),
    use_cache: bool = True,
) -> tuple[list[list[float]], list[float]]:
    """
    Runs training once per learning rate on MNIST and saves a
    comparison plot.

    Each run starts from a freshly-initialized model with the same
    seed so that differences in the curves are caused by the learning
    rate alone.

    Arguments:
        learning_rates -- learning rates to compare
                          (default: [0.01, 0.1, 1.0])
        hidden         -- hidden layer sizes (default: [128])
        epochs         -- training epochs per run
        batch_size     -- mini-batch size
        val_size       -- validation examples carved from the
                          MNIST training split
        seed           -- random seed for data splits, init, shuffles
        save_path      -- where to write the output figure
        use_cache      -- load cached weights when available

    Returns:
        (val_accuracy_per_epoch, test_accuracy_per_lr) where
        val_accuracy_per_epoch[i] is the per-epoch validation
        accuracy for learning_rates[i] and test_accuracy_per_lr[i]
        is the final test accuracy reached by that run.
    """
    if learning_rates is None:
        learning_rates = [0.01, 0.1, 1.0]
    if hidden is None:
        hidden = [128]

    cache = ModelCache() if use_cache else None
    val_curves: list[list[float]] = []
    test_accs: list[float] = []

    for lr in learning_rates:
        config = ExperimentConfig(
            layer_sizes=tuple([784, *hidden, 10]),
            activation="relu",
            optimizer="sgd",
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss="cross_entropy",
            seed=seed,
            dataset="mnist",
            val_points=val_size,
            transforms=[{"to_array": {}}],
            metrics=["accuracy"],
        )
        model, history = get_network(config, cache=cache)
        val_curve = history["val_acc"]

        X_test, y_test = _build_test(config)
        test_pred = model(X_test).argmax(axis=1)
        test_acc = accuracy(test_pred, y_test)

        val_curves.append(val_curve)
        test_accs.append(test_acc)
        print(
            f"lr={lr}: val_acc[-1]={val_curve[-1]:.4f}  "
            f"test_acc={test_acc:.4f}"
        )

    plot_lr_experiment(
        learning_rates=learning_rates,
        val_accuracy_per_epoch=val_curves,
        test_accuracy_per_lr=test_accs,
        save_path=save_path,
    )
    print(f"Saved plot to {save_path}")
    return val_curves, test_accs
