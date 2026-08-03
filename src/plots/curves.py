"""
Training-curve plotting helper.

Plots train and/or val accuracy (or loss) curves for multiple runs on
a single Axes. Train curves use solid lines; val curves use dashed
lines. Runs sharing a name share a color.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _draw_runs(
    ax: Axes,
    runs: dict[str, dict[str, list[float]]],
    colours: Any,
) -> None:
    """
    Draw train (solid) and val (dashed) curves onto *ax*.

    One color is shared per run name.  Labels follow the pattern
    ``"{name} train"`` / ``"{name} val"``.

    Arguments:
        ax      -- Matplotlib Axes to draw on
        runs    -- maps run_name → {"train": [...], "val": [...]}
                   either key may be absent
        colours -- sequence of colors (e.g. from a colormap)
    """
    for i, (name, curves) in enumerate(runs.items()):
        colour = colours[i % len(colours)]
        if "train" in curves:
            ax.plot(
                range(1, len(curves["train"]) + 1),
                curves["train"],
                color=colour,
                linestyle="-",
                label=f"{name} train",
            )
        if "val" in curves:
            ax.plot(
                range(1, len(curves["val"]) + 1),
                curves["val"],
                color=colour,
                linestyle="--",
                label=f"{name} val",
            )


def plot_training_curves(
    runs: dict[str, dict[str, list[float]]],
    save_path: Path,
    title: str = "",
    ylabel: str = "Accuracy",
) -> None:
    """
    Plot training and/or validation curves for one or more runs.

    Arguments:
        runs      -- maps run_name → {"train": [...], "val": [...]}
                     either key may be absent
        save_path -- output path for the figure
        title     -- figure title
        ylabel    -- y-axis label
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]

    _draw_runs(ax, runs, colours)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_curve_panels(
    panels: list[dict[str, Any]],
    save_path: Path,
    figsize: tuple[float, float] = (13, 5),
) -> None:
    """
    Plot a row of training-curve panels and save to *save_path*.

    Each element of *panels* is a dict with keys:
        ``"title"``  -- panel title string
        ``"runs"``   -- maps run_name → {"train": [...], "val": [...]}
        ``"ylabel"`` -- y-axis label string

    Arguments:
        panels    -- ordered list of panel specification dicts
        save_path -- output path for the figure
        figsize   -- overall figure size passed to ``plt.subplots``
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colours = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        _draw_runs(ax, panel["runs"], colours)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
