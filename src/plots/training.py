"""
Plotting helpers for the learning-rate experiment.

Produces a line plot of validation accuracy per epoch, one curve per
learning rate, and dashed horizontal reference lines showing the final
test accuracy reached at each learning rate.

This file is provided. You should not need to edit it.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def plot_lr_experiment(
    learning_rates: Sequence[float],
    val_accuracy_per_epoch: Sequence[Sequence[float]],
    test_accuracy_per_lr: Sequence[float],
    save_path: str | Path,
) -> None:
    """
    Saves a learning-rate comparison plot to `save_path`.

    Arguments:
        learning_rates         -- list of learning rates compared in
                                  the experiment
        val_accuracy_per_epoch -- list of per-epoch validation
                                  accuracies, one inner list per
                                  learning rate
        test_accuracy_per_lr   -- final test-set accuracy reached by
                                  each learning rate
        save_path              -- output path for the figure
    """
    if not (
        len(learning_rates)
        == len(val_accuracy_per_epoch)
        == len(test_accuracy_per_lr)
    ):
        raise ValueError(
            "learning_rates, val_accuracy_per_epoch, and "
            "test_accuracy_per_lr must have the same length"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colours = plt.get_cmap("tab10").colors
    for i, (lr, val_curve, test_acc) in enumerate(
        zip(learning_rates, val_accuracy_per_epoch, test_accuracy_per_lr)
    ):
        colour = colours[i % len(colours)]
        epochs = range(1, len(val_curve) + 1)
        ax.plot(
            epochs,
            val_curve,
            color=colour,
            label=f"lr = {lr} (val)",
        )
        ax.axhline(
            test_acc,
            color=colour,
            linestyle="--",
            alpha=0.6,
            label=f"lr = {lr} (test = {test_acc:.3f})",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Validation accuracy per epoch by learning rate")
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
