"""Training-diagnostic trajectory plot."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


# (history_key, y-axis label, y-axis scale)
_METRIC_ROWS: list[tuple[str, str, str]] = [
    ("train_loss", "Loss", "log"),
    ("train_acc", "Accuracy", "linear"),
    ("train_grad_norm", "Gradient norm", "log"),
    ("train_weight_norm", "Weight norm", "log"),
    ("train_grad_weight_ratio", "Grad / weight ratio", "log"),
    ("train_update_magnitude", "Update magnitude", "log"),
]

_CONDITION_COLORS: dict[str, str] = {
    "healthy": "tab:blue",
    "lr_too_high": "tab:red",
    "lr_too_low": "tab:orange",
    "deep_bad_init": "tab:purple",
}

_CONDITION_LABELS: dict[str, str] = {
    "healthy": "healthy (lr=0.05)",
    "lr_too_high": "lr too high",
    "lr_too_low": "lr too low (lr=1e-5)",
    "deep_bad_init": "deep + bad init (W×0.1)",
}


def plot_diagnostic_trajectories(
    results: dict[str, dict[str, dict[str, list[float]]]],
    condition_names: list[str],
    save_path: Path,
) -> None:
    """
    Plot per-epoch diagnostic trajectories for multiple conditions and
    datasets on a log-scale grid.

    Arguments:
        results         -- nested dict:
                           results[dataset][condition][metric_key]
                           = list of per-epoch floats
        condition_names -- ordered list of condition keys (determines
                           plot order and legend)
        save_path       -- output file path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_names = list(results.keys())
    n_datasets = len(dataset_names)
    n_rows = len(_METRIC_ROWS)

    fig, axes = plt.subplots(
        n_rows,
        n_datasets,
        figsize=(5 * n_datasets, 2.5 * n_rows),
        squeeze=False,
    )

    for col, dataset in enumerate(dataset_names):
        for row, (metric_key, metric_label, scale) in enumerate(_METRIC_ROWS):
            ax = axes[row][col]
            for cond in condition_names:
                vals = results[dataset][cond].get(metric_key, [])
                if not vals:
                    continue
                y = [
                    v
                    if not math.isnan(v) and not math.isinf(v)
                    else float("nan")
                    for v in vals
                ]
                color = _CONDITION_COLORS.get(cond)
                label = _CONDITION_LABELS.get(cond, cond)
                ax.plot(
                    range(1, len(y) + 1),
                    y,
                    label=label,
                    color=color,
                )
            ax.set_yscale(scale)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_label)
            if row == 0:
                ax.set_title(dataset)
            if col == n_datasets - 1 and row == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
