"""
Decision-boundary plotting helpers.

Renders filled decision regions (contourf) with scattered data
points overlaid, either on a single Axes or across a grid of panels.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(
    ax: matplotlib.axes.Axes,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    resolution: int = 200,
) -> None:
    """
    Draw a filled decision-region plot on `ax`.

    Arguments:
        ax         -- Axes to draw on
        predict_fn -- maps (N, 2) array → (N,) integer class array
        X          -- data points, shape (N, 2)
        y          -- integer class labels, shape (N,)
        title      -- Axes title
        resolution -- number of grid points along each axis
    """
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Z = predict_fn(grid).reshape(resolution, resolution)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="RdYlBu",
        edgecolors="k",
        s=30,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_boundary_panels(
    panels: list[list[dict[str, Any]]],
    row_labels: list[str],
    col_labels: list[str],
    save_path: Path,
) -> None:
    """
    Create a grid figure of decision-boundary plots.

    `panels[r][c]` must contain keys: "predict_fn", "X", "y", "title".
    Row labels appear as y-axis labels on the first column; column
    labels are prepended to each cell title in the top row.

    Arguments:
        panels     -- grid of panel dicts (nrows × ncols)
        row_labels -- one label per row
        col_labels -- one label per column
        save_path  -- output path for the figure
    """
    nrows = len(panels)
    ncols = len(panels[0])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))

    if nrows == 1 and ncols == 1:
        axes_grid: list[list[matplotlib.axes.Axes]] = [[axes]]  # type: ignore[list-item]
    elif nrows == 1:
        axes_grid = [list(axes)]  # type: ignore[arg-type]
    elif ncols == 1:
        axes_grid = [[row] for row in axes]  # type: ignore[union-attr]
    else:
        axes_grid = [list(row) for row in axes]  # type: ignore[union-attr]

    for r in range(nrows):
        for c in range(ncols):
            ax = axes_grid[r][c]
            cell = panels[r][c]
            cell_title = cell["title"]
            if r == 0:
                cell_title = f"{col_labels[c]}: {cell_title}"
            plot_decision_boundary(
                ax,
                cell["predict_fn"],
                cell["X"],
                cell["y"],
                cell_title,
            )
            if c == 0:
                ax.set_ylabel(row_labels[r])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
