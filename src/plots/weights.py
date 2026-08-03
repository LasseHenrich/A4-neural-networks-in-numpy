"""
Weight-visualisation helpers for the first Linear layer.

Renders each column of a weight matrix as a small image arranged in a
grid, using a diverging colormap centred at zero.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_weight_grid(
    ax: matplotlib.axes.Axes,
    W: np.ndarray,
    image_shape: tuple[int, int] = (28, 28),
    ncols: int = 8,
    title: str = "",
) -> matplotlib.image.AxesImage:
    """
    Render columns of `W` as a grid of filter images on `ax`.

    Arguments:
        ax          -- Axes to draw on
        W           -- weight matrix, shape (input_dim, n_filters)
        image_shape -- (height, width) of each filter image
        ncols       -- number of columns in the grid
        title       -- Axes title

    Returns the AxesImage so callers can attach a colorbar.
    """
    n_filters = W.shape[1]
    nrows = ceil(n_filters / ncols)
    h, w = image_shape
    canvas = np.zeros((nrows * h, ncols * w))

    for j in range(n_filters):
        r = j // ncols
        c = j % ncols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = W[:, j].reshape(
            image_shape
        )

    vmax: float = float(np.abs(W).max())
    im = ax.imshow(
        canvas,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    for col in range(1, ncols):
        ax.axvline(col * w - 0.5, color="black", linewidth=0.5)
    for row in range(1, nrows):
        ax.axhline(row * h - 0.5, color="black", linewidth=0.5)
    ax.axis("off")
    ax.set_title(title)
    return im


def plot_weight_blocks(
    blocks: list[dict[str, Any]],
    save_path: Path,
    image_shape: tuple[int, int] = (28, 28),
) -> None:
    """
    Create one subplot per block, each showing a weight grid.

    Each element of `blocks` must contain keys: "W" (shape (784, H)),
    "H" (int), "test_acc" (float).  H must be a perfect square.

    Arguments:
        blocks      -- list of block dicts
        save_path   -- output path for the figure
        image_shape -- (height, width) passed to plot_weight_grid
    """
    n = len(blocks)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    h_img, _ = image_shape
    sides = [int(np.sqrt(block["H"])) for block in blocks]

    # Scale so each filter row/col spans ~3 px at 150 DPI.
    # Enforce a minimum panel width so narrow grids (e.g. H=4) have room
    # for their title and colorbar.
    figscale = 3 * h_img / 150
    min_panel_w = 1.8
    fig_width = sum(max(s * figscale, min_panel_w) for s in sides) + 0.5
    fig_height = max(s * figscale for s in sides) + 1.5

    min_ratio = 4
    width_ratios = [max(s, min_ratio) for s in sides]
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True,
    )
    axes_list: list[matplotlib.axes.Axes]
    if n == 1:
        axes_list = [axes]  # type: ignore[list-item]
    else:
        axes_list = list(axes)  # type: ignore[arg-type]

    for ax, block, side in zip(axes_list, blocks, sides):
        H: int = block["H"]
        title: str = block.get(  # type: ignore[assignment]
            "title", f"H={H}, acc={block['test_acc']:.3f}"
        )
        im = plot_weight_grid(
            ax,
            block["W"],
            image_shape,
            ncols=side,
            title=title,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("weight", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # tight_layout replaced by constrained_layout=True on the figure.
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
