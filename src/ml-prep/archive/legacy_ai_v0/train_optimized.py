"""
ai-v0 Optimized LightGBM with:
1. Feature Engineering (interaction terms)
2. Optuna hyperparameter optimization
3. Focus on maximizing ROC-AUC (improves both classes)
"""
import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts_optimized")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and engineered features based on domain knowledge."""
    df = df.copy()
    
    # Angle ratio - key geometric relationship
    df["ANGLE_RATIO"] = df["SHOULDER-ANGLE-OUT"] / (df["SHOULDER-ANGLE-IN"] + 1e-8)
    
    # Angle difference
    df["ANGLE_DIFF"] = df["SHOULDER-ANGLE-OUT"] - df["SHOULDER-ANGLE-IN"]
    
    # Angle sum (total shoulder angle)
    df["ANGLE_SUM"] = df["SHOULDER-ANGLE-OUT"] + df["SHOULDER-ANGLE-IN"]
    
    # Diameter-Height ratio (aspect ratio)
    df["ASPECT_RATIO"] = df["DIAMETER"] / (df["HEIGHT"] + 1e-8)
    
    # Crown dimensions ratio
    df["CROWN_RATIO"] = df["CROWN-D-H"] / (df["CROWN-D-L"] + 1e-8)
    
    # Foot spread
    df["FOOT_SPREAD"] = df["FOOT-OUT"] - df["FOOT-IN"]
    
    # Key interaction: angle * diameter
    df["ANGLE_OUT_x_DIAMETER"] = df["SHOULDER-ANGLE-OUT"] * df["DIAMETER"]
    df["ANGLE_IN_x_DIAMETER"] = df["SHOULDER-ANGLE-IN"] * df["DIAMETER"]
    
    # Stroke ratio
    df["STROKE_RATIO"] = df["STROKE-OUT"] / (df["STROKE-CENTER"] + 1e-8)
    
    # Deviation from baseline (ratio=1.0)
    for col in ["SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN", "DIAMETER", "HEIGHT"]:
        df[f"{col}_DEV"] = abs(df[col] - 1.0)
    
    # Total deviation score
    df["TOTAL_DEV"] = (
        df["SHOULDER-ANGLE-OUT_DEV"] + 
        df["SHOULDER-ANGLE-IN_DEV"] + 
        df["DIAMETER_DEV"] + 
        df["HEIGHT_DEV"]
    )
    
    return df


def split_dataset(data: pd.DataFrame, label_column: str, seed: int):
    features = data.drop(columns=[label_column])
    labels = data[label_column]

    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def evaluate_model(probas, y_true, threshold=0.5):
    preds = (probas >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision_safe": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "precision_unsafe": precision_score(y_true, preds, pos_label=0, zero_division=0),
        "recall_safe": recall_score(y_true, preds, pos_label=1),
        "recall_unsafe": recall_score(y_true, preds, pos_label=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probas),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_balanced_threshold(probas, y_true):
    """Find threshold that balances both class recalls."""
    best_threshold = 0.5
    best_balance = 0
    
    for thresh in np.arange(0.35, 0.65, 0.01):
        preds = (probas >= thresh).astype(int)
        rec_0 = recall_score(y_true, preds, pos_label=0)
        rec_1 = recall_score(y_true, preds, pos_label=1)
        
        # Harmonic mean of both recalls (balanced metric)
        if rec_0 > 0 and rec_1 > 0:
            balance = 2 * rec_0 * rec_1 / (rec_0 + rec_1)
            if balance > best_balance:
                best_balance = balance
                best_threshold = thresh
    
    return best_threshold


def objective(trial, x_train, y_train, x_val, y_val):
    """Optuna objective: maximize ROC-AUC."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
    }
    
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    
    val_probas = model.predict(x_val)
    return roc_auc_score(y_val, val_probas)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--label", type=str, default="safe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("AI-v0 Optimized Training")
    print("Strategy: Feature Engineering + Optuna HPO → Maximize ROC-AUC")
    print("=" * 70)
    
    # Load and engineer features
    print("\n[1/5] Loading data and engineering features...")
    raw_data = load_dataset(args.csv)
    data = add_features(raw_data)
    
    print(f"Original features: {len(raw_data.columns) - 1}")
    print(f"After engineering: {len(data.columns) - 1}")
    new_features = set(data.columns) - set(raw_data.columns)
    print(f"New features: {sorted(new_features)}")
    
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        data, args.label, args.seed
    )
    
    # Optuna HPO
    print(f"\n[2/5] Running Optuna optimization ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_val, y_val),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best params
    print("\n[3/5] Training final model with best parameters...")
    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "random_state": 42,
        **study.best_params,
    }
    
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    
    # Find balanced threshold
    print("\n[4/5] Finding balanced threshold...")
    val_probas = model.predict(x_val)
    balanced_threshold = find_balanced_threshold(val_probas, y_val)
    print(f"Balanced threshold: {balanced_threshold:.2f}")
    
    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    test_probas = model.predict(x_test)
    
    metrics_default = evaluate_model(test_probas, y_test, threshold=0.5)
    metrics_balanced = evaluate_model(test_probas, y_test, threshold=balanced_threshold)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<22} | {'Default (0.5)':>14} | {'Balanced ({:.2f})':>14}".format(balanced_threshold))
    print("-" * 56)
    print(f"{'Accuracy':<22} | {metrics_default['accuracy']:>14.4f} | {metrics_balanced['accuracy']:>14.4f}")
    print(f"{'Recall (safe)':<22} | {metrics_default['recall_safe']:>14.4f} | {metrics_balanced['recall_safe']:>14.4f}")
    print(f"{'Recall (unsafe)':<22} | {metrics_default['recall_unsafe']:>14.4f} | {metrics_balanced['recall_unsafe']:>14.4f}")
    print(f"{'F1 Score':<22} | {metrics_default['f1']:>14.4f} | {metrics_balanced['f1']:>14.4f}")
    print(f"{'ROC-AUC':<22} | {metrics_default['roc_auc']:>14.4f} | {metrics_balanced['roc_auc']:>14.4f}")
    
    # Feature importance
    print("\n--- Top 10 Feature Importance ---")
    importance = pd.DataFrame({
        "feature": x_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # Save
    args.artifacts.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.artifacts / "lgbm_optimized.txt"))
    
    with (args.artifacts / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "best_optuna_auc": study.best_value,
            "best_params": study.best_params,
            "balanced_threshold": balanced_threshold,
            "test_default": metrics_default,
            "test_balanced": metrics_balanced,
        }, f, indent=2)
    
    with (args.artifacts / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(x_train.columns), f, indent=2)
    
    importance.to_csv(args.artifacts / "feature_importance.csv", index=False)
    
    print(f"\nArtifacts saved to {args.artifacts}/")
    print("Training complete!")


if __name__ == "__main__":
    main()
