"""Train CAD-gate model with Optuna-based AutoML search.

Outputs are compatible with `src/v2/cad_gate.py`:
  - model.joblib
  - scaler.joblib
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _bootstrap_v2(project_root: Path) -> None:
    pkg_dir = project_root / "src" / "v2"
    spec = importlib.util.spec_from_file_location(
        "v2",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to bootstrap v2 package")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2"] = mod
    spec.loader.exec_module(mod)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CAD-gate classifier with Optuna AutoML")
    p.add_argument("--dataset", default="src/ml-prep/data/cad_gate_dataset.csv")
    p.add_argument("--model-dir", default="src/ml-prep/models/cad_gate_model")
    p.add_argument("--label-col", default="label")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument(
        "--cv-n-jobs",
        type=int,
        default=0,
        help="Parallel jobs for CV scoring. 0=auto, -1=all cores.",
    )
    p.add_argument(
        "--optuna-n-jobs",
        type=int,
        default=0,
        help="Parallel jobs for Optuna trials. 0=auto.",
    )
    p.add_argument(
        "--tree-n-jobs",
        type=int,
        default=1,
        help="n_jobs for RandomForest/ExtraTrees (use 1 when CV/Optuna are parallel).",
    )
    p.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="Threads for OMP/MKL/OPENBLAS/BLIS/NUMEXPR. <=0 keeps current env.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--study-name", default="cad_gate_automl")
    return p.parse_args()


def _set_blas_threads(threads: int) -> None:
    if threads <= 0:
        return
    value = str(int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = value


def _resolve_parallel_jobs(
    *,
    cpu_count: int,
    folds: int,
    cv_n_jobs_arg: int,
    optuna_n_jobs_arg: int,
) -> tuple[int, int]:
    if cv_n_jobs_arg == 0:
        cv_n_jobs = min(max(1, folds), max(1, cpu_count))
    else:
        cv_n_jobs = int(cv_n_jobs_arg)

    if cv_n_jobs == -1:
        cv_effective = max(1, folds)
    else:
        cv_effective = max(1, cv_n_jobs)

    if optuna_n_jobs_arg == 0:
        optuna_n_jobs = max(1, cpu_count // cv_effective)
    else:
        optuna_n_jobs = int(optuna_n_jobs_arg)

    return cv_n_jobs, max(1, optuna_n_jobs)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_feature_order(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    if not rows:
        raise RuntimeError("Dataset is empty")
    keys = list(rows[0].keys())
    missing = [k for k in preferred if k not in keys]
    if missing:
        raise RuntimeError(
            "Dataset is missing required CAD-gate features: " + ", ".join(missing)
        )
    return list(preferred)


def _parse_dataset(rows: list[dict[str, str]], features: list[str], label_col: str):
    import numpy as np

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for row in rows:
        try:
            y = int(float(row[label_col]))
        except Exception:
            continue
        if y not in (0, 1):
            continue

        feat_vec: list[float] = []
        ok = True
        for name in features:
            raw = row.get(name)
            try:
                v = float(raw) if raw is not None else float("nan")
            except Exception:
                ok = False
                break
            if not math.isfinite(v):
                ok = False
                break
            feat_vec.append(v)
        if not ok:
            continue

        x_rows.append(feat_vec)
        y_rows.append(y)

    if not x_rows:
        raise RuntimeError("No valid rows after parsing dataset")
    x = np.asarray(x_rows, dtype="float64")
    y = np.asarray(y_rows, dtype="int64")
    return x, y


def _build_estimator_from_trial(trial, seed: int, tree_n_jobs: int):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_type = trial.suggest_categorical(
        "model_type",
        ["logreg", "random_forest", "extra_trees", "hist_gbdt"],
    )

    if model_type == "logreg":
        c = trial.suggest_float("logreg_c", 1.0e-3, 30.0, log=True)
        return LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 900),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            random_state=seed,
            n_jobs=int(tree_n_jobs),
            class_weight="balanced_subsample",
        )

    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 200, 1000),
            max_depth=trial.suggest_int("et_max_depth", 3, 24),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            random_state=seed,
            n_jobs=int(tree_n_jobs),
            class_weight="balanced",
        )

    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("hgb_learning_rate", 1.0e-3, 0.2, log=True),
        max_depth=trial.suggest_int("hgb_max_depth", 2, 14),
        max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 15, 255),
        min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 60),
        l2_regularization=trial.suggest_float("hgb_l2", 1.0e-8, 1.0, log=True),
        random_state=seed,
    )


def _build_estimator_from_best(best_params: dict[str, Any], seed: int, tree_n_jobs: int):
    class _FrozenTrial:
        def __init__(self, params: dict[str, Any]):
            self._params = params

        def suggest_categorical(self, name, choices):
            return self._params[name]

        def suggest_float(self, name, low, high, log=False):
            return float(self._params[name])

        def suggest_int(self, name, low, high):
            return int(self._params[name])

    trial = _FrozenTrial(best_params)
    return _build_estimator_from_trial(trial, seed, tree_n_jobs)


def _optimize_threshold(y_true, y_score):
    import numpy as np
    from sklearn.metrics import f1_score

    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_score >= float(t)).astype("int64")
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1}
    return best


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]
    _bootstrap_v2(project_root)
    _set_blas_threads(int(args.blas_threads))

    try:
        import joblib
        import numpy as np
        import optuna
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing training dependency. Install: numpy scikit-learn optuna joblib"
        ) from exc

    from v2.cad_gate import AI_V0_FEATURE_NAMES

    dataset_path = project_root / args.dataset
    model_dir = project_root / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(dataset_path)
    feature_order = _pick_feature_order(rows, AI_V0_FEATURE_NAMES)
    x, y = _parse_dataset(rows, feature_order, args.label_col)

    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if positive == 0 or negative == 0:
        raise RuntimeError(f"Dataset must include both classes. positive={positive}, negative={negative}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    folds = max(2, int(args.cv_folds))
    cpu_count = max(1, int(os.cpu_count() or 1))
    cv_n_jobs, optuna_n_jobs = _resolve_parallel_jobs(
        cpu_count=cpu_count,
        folds=folds,
        cv_n_jobs_arg=int(args.cv_n_jobs),
        optuna_n_jobs_arg=int(args.optuna_n_jobs),
    )
    print(
        "[parallel] "
        f"cpu={cpu_count} "
        f"cv_folds={folds} "
        f"cv_n_jobs={cv_n_jobs} "
        f"optuna_n_jobs={optuna_n_jobs} "
        f"tree_n_jobs={args.tree_n_jobs} "
        f"blas_threads={args.blas_threads}"
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)

    def _objective(trial: optuna.Trial) -> float:
        est = _build_estimator_from_trial(
            trial,
            seed=args.seed,
            tree_n_jobs=int(args.tree_n_jobs),
        )
        scores = cross_val_score(
            est,
            x_train_scaled,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=cv_n_jobs,
            error_score="raise",
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", study_name=args.study_name, sampler=sampler)
    study.optimize(
        _objective,
        n_trials=int(args.trials),
        timeout=int(args.timeout_sec),
        n_jobs=optuna_n_jobs,
    )

    best_est = _build_estimator_from_best(
        study.best_trial.params,
        seed=args.seed,
        tree_n_jobs=int(args.tree_n_jobs),
    )
    best_est.fit(x_train_scaled, y_train)

    y_score = best_est.predict_proba(x_test_scaled)[:, 1]
    threshold_info = _optimize_threshold(y_test, y_score)
    best_threshold = threshold_info["threshold"]
    y_pred_default = (y_score >= 0.5).astype("int64")
    y_pred_tuned = (y_score >= best_threshold).astype("int64")

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "accuracy_default_0_5": float(accuracy_score(y_test, y_pred_default)),
        "f1_default_0_5": float(f1_score(y_test, y_pred_default, zero_division=0)),
        "precision_default_0_5": float(precision_score(y_test, y_pred_default, zero_division=0)),
        "recall_default_0_5": float(recall_score(y_test, y_pred_default, zero_division=0)),
        "f1_tuned": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
        "precision_tuned": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
        "recall_tuned": float(recall_score(y_test, y_pred_tuned, zero_division=0)),
        "brier": float(brier_score_loss(y_test, y_score)),
        "recommended_threshold": float(best_threshold),
    }

    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    meta_path = model_dir / "metadata.json"
    feature_path = model_dir / "feature_order.json"

    joblib.dump(best_est, model_path)
    joblib.dump(scaler, scaler_path)
    feature_path.write_text(json.dumps(feature_order, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "class_balance": {"positive": positive, "negative": negative},
        "optuna": {
            "best_value_cv_roc_auc": float(study.best_value),
            "best_params": study.best_trial.params,
            "n_trials": len(study.trials),
        },
        "parallel": {
            "cpu_count": cpu_count,
            "cv_folds": folds,
            "cv_n_jobs": cv_n_jobs,
            "optuna_n_jobs": optuna_n_jobs,
            "tree_n_jobs": int(args.tree_n_jobs),
            "blas_threads": int(args.blas_threads),
        },
        "metrics": metrics,
        "feature_order": feature_order,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(model_path)
    print(scaler_path)
    print(meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
