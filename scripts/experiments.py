"""CLI entry point for all A4 experiment demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_NO_CACHE_HELP = "Ignore cached models and force retraining."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A4 experiment demos.")
    sub = parser.add_subparsers(dest="command", required=True)

    # lr_sweep (task j)
    p = sub.add_parser("lr_sweep", help="Task J: learning-rate sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/lr_experiment.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # boundaries (task k)
    p = sub.add_parser("boundaries", help="Task K: decision boundaries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/decision_boundaries.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # diagnostics (task l)
    p = sub.add_parser("diagnostics", help="Task L: training diagnostics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/training_diagnostics.png"),
    )

    # regression (task m)
    p = sub.add_parser("regression", help="Task M: regression fit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/regression_fit.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # scaling (task n)
    p = sub.add_parser("scaling", help="Task N: input scaling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/input_scaling.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # hparams (task o)
    p = sub.add_parser("hparams", help="Task O: hyperparameter experiments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/hyperparameters.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # weights (task p)
    p = sub.add_parser("weights", help="Task P: weight visualization")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument(
        "--save-baseline",
        type=Path,
        default=Path("results/figures/learned_features_baseline.png"),
    )
    p.add_argument(
        "--save-regularized",
        type=Path,
        default=Path("results/figures/learned_features_regularized.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    # weight_regsweep — regularization exploration sweep
    p = sub.add_parser(
        "weight_regsweep",
        help="Task P: L2 / dropout regularization sweep",
    )
    p.add_argument(
        "--hidden",
        type=int,
        required=True,
        help="First hidden-layer width (fixed for the sweep).",
    )
    p.add_argument(
        "--l2",
        type=str,
        default="1e-4",
        help="Comma-separated weight_decay values, e.g. 1e-4,1e-3,1e-2.",
    )
    p.add_argument(
        "--dropout",
        type=str,
        default="0",
        help="Comma-separated dropout probabilities, e.g. 0,0.2,0.5.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--save",
        type=Path,
        default=Path("results/figures/reg_sweep.png"),
    )
    p.add_argument(
        "--no-cache", action="store_true", default=False, help=_NO_CACHE_HELP
    )

    args = parser.parse_args()

    # Import inside main to keep module-level imports minimal
    from experiments.boundaries import (  # noqa: E402
        run_boundary_experiment,
    )
    from experiments.diagnostics import (  # noqa: E402
        run_diagnostics_experiment,
    )
    from experiments.hyperparameters import (  # noqa: E402
        run_hyperparameter_experiment,
    )
    from experiments.regression import run_regression_experiment  # noqa: E402
    from experiments.scaling import run_scaling_experiment  # noqa: E402
    from experiments.weight_viz import (  # noqa: E402
        run_reg_sweep,
        run_weight_viz_experiment,
    )
    from experiments.lr_sweep import run_lr_experiment  # noqa: E402

    if args.command == "lr_sweep":
        run_lr_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
            use_cache=not args.no_cache,
        )
    elif args.command == "boundaries":
        run_boundary_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
            use_cache=not args.no_cache,
        )
    elif args.command == "diagnostics":
        run_diagnostics_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
        )
    elif args.command == "regression":
        run_regression_experiment(
            data_seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
            use_cache=not args.no_cache,
        )
    elif args.command == "scaling":
        run_scaling_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
        )
    elif args.command == "hparams":
        run_hyperparameter_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
            use_cache=not args.no_cache,
        )
    elif args.command == "weights":
        run_weight_viz_experiment(
            seed=args.seed,
            epochs=args.epochs,
            save_path_baseline=args.save_baseline,
            save_path_regularized=args.save_regularized,
            use_cache=not args.no_cache,
        )
    elif args.command == "weight_regsweep":
        weight_decays = tuple(float(v) for v in args.l2.split(","))
        dropouts = tuple(float(v) for v in args.dropout.split(","))
        run_reg_sweep(
            H=args.hidden,
            weight_decays=weight_decays,
            dropouts=dropouts,
            seed=args.seed,
            epochs=args.epochs,
            save_path=args.save,
            use_cache=not args.no_cache,
        )


if __name__ == "__main__":
    main()
