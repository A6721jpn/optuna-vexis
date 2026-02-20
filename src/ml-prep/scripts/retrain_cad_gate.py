"""One-shot CAD-gate retraining pipeline.

1) Generate labels with v2 geometry checks (relative constraints enabled)
2) Train CAD gate model with Optuna AutoML
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain CAD gate model end-to-end")
    p.add_argument("--config", default="config/optimizer_config.yaml")
    p.add_argument("--limits", default="config/v2_limitations.yaml")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", choices=("uniform", "lhs"), default="lhs")
    p.add_argument(
        "--dataset-workers",
        type=int,
        default=1,
        help="Parallel worker processes for FreeCAD dataset labeling.",
    )
    p.add_argument(
        "--dataset-progress-every",
        type=int,
        default=50,
        help="Progress print interval during dataset labeling.",
    )
    p.add_argument("--dataset", default="src/ml-prep/data/cad_gate_dataset.csv")
    p.add_argument("--model-dir", default="src/ml-prep/models/cad_gate_model")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument(
        "--cv-n-jobs",
        type=int,
        default=0,
        help="Parallel jobs for CV scoring in trainer. 0=auto, -1=all cores.",
    )
    p.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=0,
        help="Parallel jobs for Optuna trials in trainer. 0=auto.",
    )
    p.add_argument(
        "--tree-n-jobs",
        type=int,
        default=1,
        help="n_jobs for RandomForest/ExtraTrees in trainer.",
    )
    p.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="Threads for OMP/MKL/OPENBLAS/BLIS/NUMEXPR in trainer. <=0 keeps env.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]

    gen_cmd = [
        sys.executable,
        str(project_root / "src" / "ml-prep" / "scripts" / "generate_cad_gate_dataset.py"),
        "--config",
        args.config,
        "--limits",
        args.limits,
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--sampler",
        args.sampler,
        "--workers",
        str(args.dataset_workers),
        "--progress-every",
        str(args.dataset_progress_every),
        "--output",
        args.dataset,
    ]

    train_cmd = [
        sys.executable,
        str(project_root / "src" / "ml-prep" / "scripts" / "train_cad_gate_automl.py"),
        "--dataset",
        args.dataset,
        "--model-dir",
        args.model_dir,
        "--trials",
        str(args.trials),
        "--timeout-sec",
        str(args.timeout_sec),
        "--seed",
        str(args.seed),
        "--cv-n-jobs",
        str(args.cv_n_jobs),
        "--optuna-n-jobs",
        str(args.optuna_n_jobs),
        "--tree-n-jobs",
        str(args.tree_n_jobs),
        "--blas-threads",
        str(args.blas_threads),
    ]

    print("[1/2] Generating dataset")
    subprocess.run(gen_cmd, cwd=str(project_root), check=True)
    print("[2/2] Training model")
    subprocess.run(train_cmd, cwd=str(project_root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
