"""
Scaling × learning-rate diagnostic grid plot.

Produces a 3-row × 3-column figure, one column per input-scaling
strategy. Rows show validation accuracy, training gradient norm, and
training update magnitude.  The update-magnitude row is shaded to
highlight the healthy 1e-3 to 1e-2 band.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_HEALTHY_LOW = 1e-3
_HEALTHY_HIGH = 1e-2

_ROW_KEYS = ["val_acc", "grad_norm", "update_magnitude"]
_ROW_LABELS = ["Val accuracy", "Train grad norm", "Train update mag."]
_ROW_LOG = [False, True, True]


def plot_scaling_lr_grid(
    results: dict[str, dict[float, dict[str, list[float]]]],
    save_path: Path,
) -> None:
    """
    Plot a 3-row × n-column diagnostic grid for the input-scaling
    experiment.

    Arguments:
        results   -- maps scaling_name → {lr → {metric → values}}
                     where metric is one of "val_acc", "grad_norm",
                     "update_magnitude"
        save_path -- output path for the figure
    """
    scaling_names = list(results.keys())
    n_cols = len(scaling_names)
    n_rows = len(_ROW_KEYS)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        sharey="row",
        squeeze=False,
    )

    # Colour-map: one colour per learning rate, consistent across panels.
    all_lrs = list(next(iter(results.values())).keys())
    cmap = plt.get_cmap("tab10")  # type: ignore[attr-defined]
    colours = {lr: cmap(i) for i, lr in enumerate(all_lrs)}

    for col, name in enumerate(scaling_names):
        lr_curves = results[name]
        for row, (metric_key, y_label, use_log) in enumerate(
            zip(_ROW_KEYS, _ROW_LABELS, _ROW_LOG)
        ):
            ax = axes[row][col]

            for lr, metric_dict in lr_curves.items():
                values = metric_dict.get(metric_key, [])
                if not values:
                    continue
                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    label=f"lr={lr}",
                    color=colours[lr],
                )

            if use_log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1)

            # Shade healthy update-magnitude band on the bottom row.
            if metric_key == "update_magnitude":
                ax.axhspan(
                    _HEALTHY_LOW,
                    _HEALTHY_HIGH,
                    alpha=0.12,
                    color="green",
                    label="healthy band",
                )

            # Column title on the top row only.
            if row == 0:
                ax.set_title(name)

            # y-label on the leftmost column only.
            if col == 0:
                ax.set_ylabel(y_label)

            # x-label on the bottom row only.
            if row == n_rows - 1:
                ax.set_xlabel("Epoch")

            # Legend only on top-left panel (to avoid clutter).
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
