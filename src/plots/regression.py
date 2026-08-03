"""
Regression-fit plotting helpers.

Overlays scattered training data, a true function curve, and a
predicted function curve on a single Axes, or across a row of panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_regression_fit(
    ax: matplotlib.axes.Axes,
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    """
    Overlay training scatter, true curve, and predicted curve on `ax`.

    Arguments:
        ax      -- Axes to draw on
        X_train -- training inputs, shape (N,) or (N, 1)
        y_train -- training targets, shape (N,) or (N, 1)
        x_grid  -- evaluation grid, shape (M,) or (M, 1)
        y_true  -- true function values on the grid, same shape as x_grid
        y_pred  -- predicted values on the grid, same shape as x_grid
        title   -- Axes title
    """
    ax.scatter(
        X_train.ravel(),
        y_train.ravel(),
        s=15,
        zorder=3,
        label="train",
    )
    ax.plot(
        x_grid.ravel(),
        y_true.ravel(),
        linestyle="--",
        label="true",
    )
    ax.plot(
        x_grid.ravel(),
        y_pred.ravel(),
        label="predicted",
    )
    ax.set_title(title)
    ax.legend()


def plot_regression_panels(
    panels: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """
    Create a row of regression-fit subplots, one per panel.

    Each element of `panels` must contain keys: "X_train", "y_train",
    "x_grid", "y_true", "y_pred", "title".

    Arguments:
        panels    -- list of panel dicts
        save_path -- output path for the figure
    """
    n = len(panels)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes_list: list[matplotlib.axes.Axes]
    if n == 1:
        axes_list = [axes]  # type: ignore[list-item]
    else:
        axes_list = list(axes)  # type: ignore[arg-type]

    for ax, panel in zip(axes_list, panels):
        plot_regression_fit(
            ax,
            panel["X_train"],
            panel["y_train"],
            panel["x_grid"],
            panel["y_true"],
            panel["y_pred"],
            panel["title"],
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
