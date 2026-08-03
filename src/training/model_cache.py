"""Content-addressed model cache for experiment reproducibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nn import MLP, Module
from nn.linear import Linear


_CACHE_DIR = Path("results/models")

_TRAINING_CRITICAL_FILES = [
    "src/nn/module.py",
    "src/nn/linear.py",
    "src/nn/activations.py",
    "src/nn/loss.py",
    "src/nn/sequential.py",
    "src/nn/mlp.py",
    "src/nn/dropout.py",
    "src/optim/optimizer.py",
    "src/optim/sgd.py",
    "src/optim/adam.py",
    "src/training/train_ffnn.py",
    "src/training/get_network.py",
]


@dataclass
class ExperimentConfig:
    # Architecture
    layer_sizes: tuple[int, ...]
    activation: str = "relu"  # "relu" | "leaky_relu" | "gelu"
    dropout: float = 0.0
    weight_scale: float = 1.0

    # Optimizer
    optimizer: str = "sgd"  # "sgd" | "adam"
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Training
    epochs: int = 1
    batch_size: int = 1
    loss: str = "cross_entropy"  # "mse" | "cross_entropy"
    seed: int = 0

    # Data
    dataset: str = "mnist"
    n_train: int | None = None
    n_samples: int | None = None
    noise: float | None = None
    data_seed: int | None = None
    val_points: int | None = None

    # Transforms — ordered list of single-entry dicts: {name: params_dict}
    transforms: list[dict[str, dict[str, Any]]] = field(default_factory=list)

    # Metrics tracked during training (e.g. ["accuracy"])
    metrics: list[str] = field(default_factory=list)


def _code_hash(root: Path = Path(".")) -> str:
    if (root / ".reference_mode").exists():
        return "REFR"
    h = hashlib.sha256()
    for rel in _TRAINING_CRITICAL_FILES:
        p = root / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def _config_hash(config: ExperimentConfig) -> str:
    d = asdict(config)
    d.pop("metrics")  # metrics are checked separately as a subset, not hashed
    cfg_blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(cfg_blob).hexdigest()[:4]


def cache_key(config: ExperimentConfig, root: Path = Path(".")) -> str:
    """Returns '{cfg4}-{code4}' identifying this config + src revision."""
    return f"{_config_hash(config)}-{_code_hash(root)[:4]}"


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None


def _build_mlp(config: ExperimentConfig) -> MLP:
    return MLP(
        list(config.layer_sizes),
        config.activation,
        dropout=config.dropout,
        seed=config.seed,
    )


class ModelCache:
    """
    Content-addressed store for trained MLP weights.

    Each entry is keyed by a 4-char hash of the ExperimentConfig and a
    4-char hash of training-critical source files.  Any edit to those
    files automatically invalidates existing cache entries.

    Storage layout::

        results/models/<iso8601>-<cfg4>-<code4>/
            config.json    -- ExperimentConfig as a dict
            weights.npz    -- W_i, b_i for each Linear layer in order
            history.json   -- train_ffnn return value (may be {})
            metadata.json  -- cfg_hash, code_hash, git_sha, created_at
    """

    def __init__(self, cache_dir: Path = _CACHE_DIR) -> None:
        self._dir = cache_dir

    def _hashes(self, config: ExperimentConfig) -> tuple[str, str]:
        """Returns (cfg4, code4) for this config + current src."""
        return _config_hash(config), _code_hash()[:4]

    def _find(self, config: ExperimentConfig) -> Path | None:
        cfg4, code4 = self._hashes(config)
        matches = sorted(self._dir.glob(f"*-{cfg4}-{code4}"))
        requested = set(config.metrics)
        for entry in reversed(matches):
            cached = json.loads((entry / "config.json").read_text())
            if requested.issubset(set(cached.get("metrics", []))):
                return entry
        return None

    def has(self, config: ExperimentConfig) -> bool:
        return self._find(config) is not None

    def save(
        self,
        config: ExperimentConfig,
        model: MLP,
        history: dict[str, list[float]] | None = None,
    ) -> None:
        cfg4, code4 = self._hashes(config)
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        entry = self._dir / f"{ts}-{cfg4}-{code4}"
        entry.mkdir(parents=True, exist_ok=True)

        # config
        (entry / "config.json").write_text(
            json.dumps(asdict(config), indent=2)
        )

        # weights — index over Linear layers only
        weights: dict[str, np.ndarray] = {}
        i = 0
        for layer in model.net.layers:
            if isinstance(layer, Linear):
                weights[f"W_{i}"] = layer.W
                weights[f"b_{i}"] = layer.b
                i += 1
        np.savez(entry / "weights.npz", **weights)

        # history
        (entry / "history.json").write_text(
            json.dumps(history or {}, indent=2)
        )

        # metadata
        meta: dict[str, Any] = {
            "cfg_hash": cfg4,
            "code_hash": code4,
            "git_sha": _git_sha(),
            "created_at": ts,
        }
        (entry / "metadata.json").write_text(json.dumps(meta, indent=2))

    def load(
        self, config: ExperimentConfig
    ) -> tuple[MLP, dict[str, list[float]]]:
        entry = self._find(config)
        if entry is None:
            raise KeyError(f"No cached model for config: {config}")

        model = _build_mlp(config)
        weights = np.load(entry / "weights.npz")
        i = 0
        for layer in model.net.layers:
            if isinstance(layer, Linear):
                np.copyto(layer.W, weights[f"W_{i}"])
                np.copyto(layer.b, weights[f"b_{i}"])
                i += 1

        history: dict[str, list[float]] = json.loads(
            (entry / "history.json").read_text()
        )
        return model, history
