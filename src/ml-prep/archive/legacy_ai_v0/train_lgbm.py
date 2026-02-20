"""
ai-v0 LightGBM training script with aggressive unsafe detection.
Focus: Maximize unsafe recall while maintaining reasonable precision.
"""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
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
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts_lgbm")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


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


def find_optimal_threshold_for_unsafe(probas, y_true, target_unsafe_recall=0.90):
    """
    Find threshold that achieves target unsafe recall.
    Higher threshold = more samples classified as safe = LOWER unsafe recall.
    So for higher unsafe recall, we need HIGHER threshold (more aggressive unsafe detection).
    """
    best_threshold = 0.5
    best_score = 0
    
    print("\n=== Threshold Optimization (Targeting High Unsafe Recall) ===")
    print(f"{'Thresh':>7} | {'Acc':>6} | {'Prec_0':>7} | {'Rec_0':>6} | {'Prec_1':>7} | {'Rec_1':>6} | {'F1':>6}")
    print("-" * 70)
    
    for thresh in np.arange(0.40, 0.85, 0.05):
        preds = (probas >= thresh).astype(int)
        
        acc = accuracy_score(y_true, preds)
        prec_0 = precision_score(y_true, preds, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_true, preds, pos_label=0)
        prec_1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_true, preds, pos_label=1)
        f1 = f1_score(y_true, preds)
        
        print(f"{thresh:>7.2f} | {acc:>6.3f} | {prec_0:>7.3f} | {rec_0:>6.3f} | {prec_1:>7.3f} | {rec_1:>6.3f} | {f1:>6.3f}")
        
        # Weighted score prioritizing unsafe recall heavily
        # We want unsafe recall >= 0.85 ideally
        weighted = 0.7 * rec_0 + 0.2 * prec_0 + 0.1 * f1
        
        if weighted > best_score:
            best_score = weighted
            best_threshold = thresh
    
    print("-" * 70)
    print(f"Selected threshold: {best_threshold:.2f}")
    
    return best_threshold


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
        "classification_report": classification_report(y_true, preds, zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM classifier with unsafe focus.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--label", type=str, default="safe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("AI-v0 LightGBM Training")
    print("Focus: Aggressive Unsafe Detection")
    print("=" * 70)
    
    data = load_dataset(args.csv)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        data, args.label, args.seed
    )
    
    # Calculate class weights
    n_unsafe = (y_train == 0).sum()
    n_safe = (y_train == 1).sum()
    
    # Higher weight for unsafe class to boost its importance
    scale_pos_weight = n_unsafe / n_safe  # This is inverted for LightGBM
    
    print(f"\nClass distribution: unsafe={n_unsafe}, safe={n_safe}")
    print(f"Scale factor: {n_safe/n_unsafe:.2f}x weight on unsafe class")
    
    # LightGBM parameters optimized for unsafe detection
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": args.seed,
        "verbose": -1,
        # Heavy class weight on unsafe (class 0)
        # Note: scale_pos_weight > 1 gives more weight to positive class (safe)
        # To boost unsafe detection, we use weight < 1 or handle via threshold
        "scale_pos_weight": n_unsafe / n_safe,  # < 1 to boost unsafe class importance
    }
    
    print("\n[1/4] Training LightGBM with class weighting...")
    
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=100),
        ],
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    
    print("\n[2/4] Finding optimal threshold...")
    val_probas = model.predict(x_val)
    optimal_threshold = find_optimal_threshold_for_unsafe(val_probas, y_val)
    
    print("\n[3/4] Feature importance...")
    importance = pd.DataFrame({
        "feature": x_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))
    
    print("\n[4/4] Evaluating on test set...")
    test_probas = model.predict(x_test)
    
    metrics_baseline = evaluate_model(test_probas, y_test, threshold=0.5)
    metrics_optimized = evaluate_model(test_probas, y_test, threshold=optimal_threshold)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<22} | {'Baseline (0.5)':>15} | {'Optimized ({:.2f})':>15}".format(optimal_threshold))
    print("-" * 58)
    print(f"{'Accuracy':<22} | {metrics_baseline['accuracy']:>15.4f} | {metrics_optimized['accuracy']:>15.4f}")
    print(f"{'Precision (safe)':<22} | {metrics_baseline['precision_safe']:>15.4f} | {metrics_optimized['precision_safe']:>15.4f}")
    print(f"{'Precision (unsafe)':<22} | {metrics_baseline['precision_unsafe']:>15.4f} | {metrics_optimized['precision_unsafe']:>15.4f}")
    print(f"{'Recall (safe)':<22} | {metrics_baseline['recall_safe']:>15.4f} | {metrics_optimized['recall_safe']:>15.4f}")
    print(f"{'Recall (unsafe)':<22} | {metrics_baseline['recall_unsafe']:>15.4f} | {metrics_optimized['recall_unsafe']:>15.4f}")
    print(f"{'F1 Score':<22} | {metrics_baseline['f1']:>15.4f} | {metrics_optimized['f1']:>15.4f}")
    print(f"{'ROC-AUC':<22} | {metrics_baseline['roc_auc']:>15.4f} | {metrics_optimized['roc_auc']:>15.4f}")
    
    # Confusion matrix for optimized
    cm = metrics_optimized["confusion_matrix"]
    print(f"\nConfusion Matrix (threshold={optimal_threshold:.2f}):")
    print(f"  Predicted:   unsafe    safe")
    print(f"  Actual unsafe: {cm['tn']:>5}   {cm['fn']:>5}")
    print(f"  Actual safe:   {cm['fp']:>5}   {cm['tp']:>5}")
    
    # Save artifacts
    args.artifacts.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.artifacts / "lgbm_model.txt"))
    
    with (args.artifacts / "metrics_lgbm.json").open("w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": optimal_threshold,
            "test_baseline": metrics_baseline,
            "test_optimized": metrics_optimized,
            "best_iteration": model.best_iteration,
        }, f, indent=2, ensure_ascii=False)
    
    with (args.artifacts / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(x_train.columns), f, indent=2)
    
    importance.to_csv(args.artifacts / "feature_importance.csv", index=False)
    
    with (args.artifacts / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": optimal_threshold, "model_type": "lightgbm"}, f, indent=2)
    
    print(f"\nArtifacts saved to {args.artifacts}/")
    print("Training complete!")


if __name__ == "__main__":
    main()
