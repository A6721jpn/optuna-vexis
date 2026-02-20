"""
ai-v0 improved training script with:
1. Class weight balancing
2. Threshold optimization based on validation set
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts_improved")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_dataset(data: pd.DataFrame, label_column: str, seed: int):
    features = data.drop(columns=[label_column])
    labels = data[label_column]

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_model_with_class_weight(x_train, y_train, seed: int) -> MLPClassifier:
    """Train MLP with class weight balancing."""
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Slightly larger network
        activation="relu",
        solver="adam",
        alpha=0.0001,  # L2 regularization
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    )
    
    # Compute sample weights for class balancing
    sample_weights = compute_sample_weight('balanced', y_train)
    
    # Note: MLPClassifier doesn't directly support sample_weight in fit()
    # We'll use a partial_fit approach or switch strategy
    # Actually, sklearn's MLPClassifier does NOT support sample_weight
    # So we'll oversample the minority class instead
    
    # Oversample minority class
    X_train_arr = np.array(x_train)
    y_train_arr = np.array(y_train)
    
    minority_mask = y_train_arr == 0
    majority_mask = y_train_arr == 1
    
    n_minority = minority_mask.sum()
    n_majority = majority_mask.sum()
    
    # Upsample minority class
    if n_minority < n_majority:
        rng = np.random.default_rng(seed)
        minority_indices = np.where(minority_mask)[0]
        upsample_indices = rng.choice(minority_indices, size=n_majority - n_minority, replace=True)
        
        X_balanced = np.vstack([X_train_arr, X_train_arr[upsample_indices]])
        y_balanced = np.concatenate([y_train_arr, y_train_arr[upsample_indices]])
        
        # Shuffle
        shuffle_idx = rng.permutation(len(y_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
    else:
        X_balanced = X_train_arr
        y_balanced = y_train_arr
    
    print(f"Original training size: {len(y_train_arr)} (minority: {n_minority}, majority: {n_majority})")
    print(f"Balanced training size: {len(y_balanced)} (each class: ~{n_majority})")
    
    model.fit(X_balanced, y_balanced)
    return model


def find_optimal_threshold(model, x_val, y_val, target_unsafe_recall: float = 0.85):
    """
    Find optimal threshold to maximize unsafe class recall while maintaining reasonable precision.
    
    For CAE guard use case, we want to minimize false negatives (unsafe predicted as safe).
    """
    probas = model.predict_proba(x_val)[:, 1]  # P(safe)
    
    # For unsafe detection, we want to reject samples with low P(safe)
    # Lower threshold = more samples classified as safe = higher unsafe recall
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    
    print("\n=== Threshold Optimization ===")
    print(f"{'Threshold':>10} | {'Acc':>6} | {'Prec':>6} | {'Rec_0':>6} | {'Rec_1':>6} | {'F1':>6}")
    print("-" * 60)
    
    for thresh in thresholds:
        preds = (probas >= thresh).astype(int)
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        recall_0 = recall_score(y_val, preds, pos_label=0)  # unsafe recall
        recall_1 = recall_score(y_val, preds, pos_label=1)  # safe recall
        f1 = f1_score(y_val, preds)
        
        print(f"{thresh:>10.2f} | {acc:>6.3f} | {prec:>6.3f} | {recall_0:>6.3f} | {recall_1:>6.3f} | {f1:>6.3f}")
        
        # Prioritize unsafe recall (recall_0) while maintaining reasonable overall performance
        # Use a weighted score: 0.6 * recall_0 + 0.4 * f1
        weighted_score = 0.6 * recall_0 + 0.4 * f1
        
        if weighted_score > best_f1:
            best_f1 = weighted_score
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'accuracy': acc,
                'precision': prec,
                'recall_unsafe': recall_0,
                'recall_safe': recall_1,
                'f1': f1,
                'weighted_score': weighted_score,
            }
    
    print("-" * 60)
    print(f"Selected threshold: {best_threshold:.2f} (weighted_score={best_f1:.3f})")
    
    return best_threshold, best_metrics


def evaluate_model(model, x, y, threshold: float = 0.5):
    probas = model.predict_proba(x)[:, 1]
    preds = (probas >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "recall_unsafe": recall_score(y, preds, pos_label=0),
        "recall_safe": recall_score(y, preds, pos_label=1),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probas),
        "classification_report": classification_report(y, preds, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved v0 safe classifier.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="safe",
        help="Label column name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Output directory for artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("AI-v0 Improved Training")
    print("Improvements: Class Balancing + Threshold Optimization")
    print("=" * 60)
    
    data = load_dataset(args.csv)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        data, args.label, args.seed
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    print("\n[1/3] Training model with class balancing...")
    model = train_model_with_class_weight(x_train_scaled, y_train, args.seed)
    
    print("\n[2/3] Finding optimal threshold on validation set...")
    optimal_threshold, threshold_metrics = find_optimal_threshold(model, x_val_scaled, y_val)
    
    print("\n[3/3] Evaluating on test set...")
    
    # Compare baseline (0.5) vs optimized threshold
    metrics_baseline = evaluate_model(model, x_test_scaled, y_test, threshold=0.5)
    metrics_optimized = evaluate_model(model, x_test_scaled, y_test, threshold=optimal_threshold)
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} | {'Baseline (0.5)':>15} | {'Optimized ({:.2f})':>15}".format(optimal_threshold))
    print("-" * 55)
    print(f"{'Accuracy':<20} | {metrics_baseline['accuracy']:>15.4f} | {metrics_optimized['accuracy']:>15.4f}")
    print(f"{'Precision':<20} | {metrics_baseline['precision']:>15.4f} | {metrics_optimized['precision']:>15.4f}")
    print(f"{'Recall (safe)':<20} | {metrics_baseline['recall_safe']:>15.4f} | {metrics_optimized['recall_safe']:>15.4f}")
    print(f"{'Recall (unsafe)':<20} | {metrics_baseline['recall_unsafe']:>15.4f} | {metrics_optimized['recall_unsafe']:>15.4f}")
    print(f"{'F1 Score':<20} | {metrics_baseline['f1']:>15.4f} | {metrics_optimized['f1']:>15.4f}")
    print(f"{'ROC-AUC':<20} | {metrics_baseline['roc_auc']:>15.4f} | {metrics_optimized['roc_auc']:>15.4f}")
    
    # Calculate improvement
    recall_unsafe_improvement = metrics_optimized['recall_unsafe'] - metrics_baseline['recall_unsafe']
    print(f"\n{'Unsafe Recall Δ':<20} | {'-':>15} | {recall_unsafe_improvement:>+15.4f}")
    
    args.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifacts / "mlp_model_improved.joblib")
    joblib.dump(scaler, args.artifacts / "scaler.joblib")

    all_metrics = {
        "optimal_threshold": optimal_threshold,
        "threshold_selection_metrics": threshold_metrics,
        "train": evaluate_model(model, x_train_scaled, y_train, threshold=optimal_threshold),
        "val": evaluate_model(model, x_val_scaled, y_val, threshold=optimal_threshold),
        "test_baseline": metrics_baseline,
        "test_optimized": metrics_optimized,
    }
    
    with (args.artifacts / "metrics_improved.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    with (args.artifacts / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(x_train.columns), f, indent=2, ensure_ascii=False)

    # Save threshold config for predict.py
    with (args.artifacts / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": optimal_threshold}, f, indent=2)

    print(f"\nArtifacts saved to {args.artifacts}/")
    print("Training complete!")


if __name__ == "__main__":
    main()
