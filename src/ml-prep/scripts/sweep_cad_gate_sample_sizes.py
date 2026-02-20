"""Record CAD-gate metrics for current model, then retrain on larger sample sizes.

Flow:
1) Read current model artifacts (expected to be the current 5k run) and append metrics to CSV.
2) Retrain with each requested sample size.
3) Append the same metrics for each retrained model to the same CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write current CAD-gate metrics, then retrain for 10k/20k and append."
    )
    p.add_argument("--config", default="config/optimizer_config.yaml")
    p.add_argument("--limits", default="config/v2_limitations.yaml")
    p.add_argument("--dataset", default="src/ml-prep/data/cad_gate_dataset.csv")
    p.add_argument("--model-dir", default="src/ml-prep/models/cad_gate_model")
    p.add_argument("--csv-output", default="src/ml-prep/reports/cad_gate_sample_size_sweep.csv")
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
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--timeout-sec", type=int, default=5400)
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
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--sizes", default="10000,20000")
    p.add_argument("--eval-thresholds", default="0.44,0.5")
    p.add_argument("--expected-baseline-samples", type=int, default=5000)
    p.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to existing CSV instead of recreating it.",
    )
    return p.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        values.append(int(s))
    if not values:
        raise RuntimeError("No sample sizes specified")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        values.append(float(s))
    return values


def _threshold_key(th: float) -> str:
    text = f"{th:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg_").replace(".", "_")


def _resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute():
        rel = project_root / p
        if rel.exists():
            return rel
    # Convert Windows absolute path (e.g., C:\repo\...) for POSIX execution.
    if len(path_str) >= 3 and path_str[1:3] == ":\\":
        drive = path_str[0].lower()
        suffix = path_str[2:].replace("\\", "/")
        posix_p = Path(f"/mnt/{drive}{suffix}")
        if posix_p.exists():
            return posix_p
    return p if p.is_absolute() else (project_root / p)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(dataset_path: Path, features: list[str]) -> tuple[Any, Any]:
    import numpy as np

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                y = int(float(row["label"]))
            except Exception:
                continue
            if y not in (0, 1):
                continue

            feat_vec: list[float] = []
            valid = True
            for name in features:
                raw = row.get(name)
                try:
                    value = float(raw) if raw is not None else float("nan")
                except Exception:
                    valid = False
                    break
                if not np.isfinite(value):
                    valid = False
                    break
                feat_vec.append(value)

            if not valid:
                continue
            x_rows.append(feat_vec)
            y_rows.append(y)

    if not x_rows:
        raise RuntimeError(f"No valid dataset rows in {dataset_path}")
    x = np.asarray(x_rows, dtype="float64")
    y = np.asarray(y_rows, dtype="int64")
    return x, y


def _rate_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    pred = (y_score >= threshold).astype("int64")
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fpr = (fp / (fp + tn)) if (fp + tn) else float("nan")
    fnr = (fn / (fn + tp)) if (fn + tp) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _collect_row(
    *,
    project_root: Path,
    model_dir: Path,
    seed: int,
    test_size: float,
    eval_thresholds: list[float],
    run_label: str,
    requested_samples: int | None,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: joblib") from exc

    meta_path = model_dir / "metadata.json"
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"

    if not meta_path.exists():
        raise RuntimeError(f"Missing metadata: {meta_path}")
    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")
    if not scaler_path.exists():
        raise RuntimeError(f"Missing scaler: {scaler_path}")

    meta = _load_json(meta_path)
    feature_order = list(meta.get("feature_order") or [])
    if not feature_order:
        raise RuntimeError(f"feature_order missing in {meta_path}")

    dataset_path = _resolve_path(str(meta.get("dataset", "")), project_root)
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset not found: {dataset_path}")

    x, y = _load_dataset(dataset_path, feature_order)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )
    # Keep y_train for parity with the training split even if unused.
    del x_train, y_train

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    y_score = model.predict_proba(scaler.transform(x_test))[:, 1]

    metrics = dict(meta.get("metrics") or {})
    recommended_threshold = float(metrics.get("recommended_threshold", 0.5))

    row: dict[str, Any] = {
        "run_label": run_label,
        "requested_samples": requested_samples if requested_samples is not None else "",
        "actual_rows": int(meta.get("n_rows", 0)),
        "positive": int((meta.get("class_balance") or {}).get("positive", 0)),
        "negative": int((meta.get("class_balance") or {}).get("negative", 0)),
        "model_timestamp": str(meta.get("timestamp", "")),
        "dataset_path": str(meta.get("dataset", "")),
        "optuna_trials": int((meta.get("optuna") or {}).get("n_trials", 0)),
        "optuna_best_cv_roc_auc": float(
            (meta.get("optuna") or {}).get("best_value_cv_roc_auc", float("nan"))
        ),
        "best_model_type": str(
            ((meta.get("optuna") or {}).get("best_params") or {}).get("model_type", "")
        ),
        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
        "pr_auc": float(metrics.get("pr_auc", float("nan"))),
        "accuracy_default_0_5": float(metrics.get("accuracy_default_0_5", float("nan"))),
        "f1_default_0_5": float(metrics.get("f1_default_0_5", float("nan"))),
        "precision_default_0_5": float(metrics.get("precision_default_0_5", float("nan"))),
        "recall_default_0_5": float(metrics.get("recall_default_0_5", float("nan"))),
        "brier": float(metrics.get("brier", float("nan"))),
        "recommended_threshold": recommended_threshold,
    }

    for th in eval_thresholds:
        key = _threshold_key(th)
        rate = _rate_metrics(y_test, y_score, float(th))
        row[f"tp_t_{key}"] = rate["tp"]
        row[f"fp_t_{key}"] = rate["fp"]
        row[f"tn_t_{key}"] = rate["tn"]
        row[f"fn_t_{key}"] = rate["fn"]
        row[f"fpr_t_{key}"] = rate["fpr"]
        row[f"fnr_t_{key}"] = rate["fnr"]

    rec_rate = _rate_metrics(y_test, y_score, recommended_threshold)
    row["tp_t_recommended"] = rec_rate["tp"]
    row["fp_t_recommended"] = rec_rate["fp"]
    row["tn_t_recommended"] = rec_rate["tn"]
    row["fn_t_recommended"] = rec_rate["fn"]
    row["fpr_t_recommended"] = rec_rate["fpr"]
    row["fnr_t_recommended"] = rec_rate["fnr"]
    return row


def _fieldnames(eval_thresholds: list[float]) -> list[str]:
    cols = [
        "run_label",
        "requested_samples",
        "actual_rows",
        "positive",
        "negative",
        "model_timestamp",
        "dataset_path",
        "optuna_trials",
        "optuna_best_cv_roc_auc",
        "best_model_type",
        "roc_auc",
        "pr_auc",
        "accuracy_default_0_5",
        "f1_default_0_5",
        "precision_default_0_5",
        "recall_default_0_5",
        "brier",
        "recommended_threshold",
    ]
    for th in eval_thresholds:
        key = _threshold_key(th)
        cols.extend(
            [
                f"tp_t_{key}",
                f"fp_t_{key}",
                f"tn_t_{key}",
                f"fn_t_{key}",
                f"fpr_t_{key}",
                f"fnr_t_{key}",
            ]
        )
    cols.extend(
        [
            "tp_t_recommended",
            "fp_t_recommended",
            "tn_t_recommended",
            "fn_t_recommended",
            "fpr_t_recommended",
            "fnr_t_recommended",
        ]
    )
    return cols


def _run_retrain(
    *,
    project_root: Path,
    config: str,
    limits: str,
    samples: int,
    seed: int,
    sampler: str,
    dataset_workers: int,
    dataset_progress_every: int,
    dataset: str,
    model_dir: str,
    trials: int,
    timeout_sec: int,
    cv_n_jobs: int,
    optuna_n_jobs: int,
    tree_n_jobs: int,
    blas_threads: int,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "src" / "ml-prep" / "scripts" / "retrain_cad_gate.py"),
        "--config",
        config,
        "--limits",
        limits,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--sampler",
        sampler,
        "--dataset-workers",
        str(dataset_workers),
        "--dataset-progress-every",
        str(dataset_progress_every),
        "--dataset",
        dataset,
        "--model-dir",
        model_dir,
        "--trials",
        str(trials),
        "--timeout-sec",
        str(timeout_sec),
        "--cv-n-jobs",
        str(cv_n_jobs),
        "--optuna-n-jobs",
        str(optuna_n_jobs),
        "--tree-n-jobs",
        str(tree_n_jobs),
        "--blas-threads",
        str(blas_threads),
    ]
    print(f"[train] samples={samples}")
    subprocess.run(cmd, cwd=str(project_root), check=True)


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]

    eval_thresholds = _parse_float_list(args.eval_thresholds)
    sample_sizes = _parse_int_list(args.sizes)
    model_dir = project_root / args.model_dir
    csv_path = project_root / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append_existing else "w"
    write_header = True
    if mode == "a" and csv_path.exists() and csv_path.stat().st_size > 0:
        write_header = False

    with csv_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(eval_thresholds))
        if write_header:
            writer.writeheader()

        baseline_row = _collect_row(
            project_root=project_root,
            model_dir=model_dir,
            seed=args.seed,
            test_size=args.test_size,
            eval_thresholds=eval_thresholds,
            run_label="baseline_current",
            requested_samples=None,
        )
        baseline_rows = int(baseline_row["actual_rows"])
        if baseline_rows != int(args.expected_baseline_samples):
            print(
                "[warn] baseline actual_rows="
                f"{baseline_rows} (expected {args.expected_baseline_samples})"
            )
        writer.writerow(baseline_row)
        f.flush()
        print(
            f"[csv] appended baseline_current rows={baseline_row['actual_rows']} "
            f"roc_auc={baseline_row['roc_auc']:.6f}"
        )

        for samples in sample_sizes:
            _run_retrain(
                project_root=project_root,
                config=args.config,
                limits=args.limits,
                samples=samples,
                seed=args.seed,
                sampler=args.sampler,
                dataset_workers=args.dataset_workers,
                dataset_progress_every=args.dataset_progress_every,
                dataset=args.dataset,
                model_dir=args.model_dir,
                trials=args.trials,
                timeout_sec=args.timeout_sec,
                cv_n_jobs=args.cv_n_jobs,
                optuna_n_jobs=args.optuna_n_jobs,
                tree_n_jobs=args.tree_n_jobs,
                blas_threads=args.blas_threads,
            )
            row = _collect_row(
                project_root=project_root,
                model_dir=model_dir,
                seed=args.seed,
                test_size=args.test_size,
                eval_thresholds=eval_thresholds,
                run_label=f"retrain_{samples}",
                requested_samples=samples,
            )
            writer.writerow(row)
            f.flush()
            print(
                f"[csv] appended retrain_{samples} rows={row['actual_rows']} "
                f"roc_auc={row['roc_auc']:.6f}"
            )

    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
