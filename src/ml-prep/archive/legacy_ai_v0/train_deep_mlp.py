"""
Deep MLP Architecture Search with Optuna.
Searches: layer count, layer sizes, learning rate, regularization.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
ARTIFACTS_DIR = Path(__file__).with_name("artifacts_deep_mlp")


def load_and_split(seed=42):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["safe"])
    y = df["safe"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    return X_train_s, X_val_s, X_test_s, y_train.values, y_val.values, y_test.values, scaler, X.columns.tolist()


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "recall_safe": recall_score(y, y_pred, pos_label=1),
        "recall_unsafe": recall_score(y, y_pred, pos_label=0),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximize ROC-AUC with deep MLP."""
    
    # Number of layers (2-5)
    n_layers = trial.suggest_int("n_layers", 2, 5)
    
    # Layer sizes (decreasing pattern)
    layers = []
    prev_size = 512
    for i in range(n_layers):
        # Each layer can be 32-512, generally decreasing
        min_size = 32
        max_size = min(prev_size, 512)
        size = trial.suggest_int(f"layer_{i}_size", min_size, max_size, step=32)
        layers.append(size)
        prev_size = size
    
    hidden_layer_sizes = tuple(layers)
    
    # Other hyperparameters
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)  # L2 regularization
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate="adaptive",
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_proba)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Deep MLP Architecture Search")
    print(f"Trials: {args.n_trials}")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = load_and_split(args.seed)
    
    print(f"\nData: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # Optuna study
    print(f"\n[1/3] Running Optuna optimization ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Reconstruct best architecture
    best_params = study.best_params
    n_layers = best_params["n_layers"]
    layers = tuple(best_params[f"layer_{i}_size"] for i in range(n_layers))
    
    print(f"Best architecture: {layers}")
    
    # Train final model
    print("\n[2/3] Training final model with best architecture...")
    final_model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        alpha=best_params["alpha"],
        learning_rate="adaptive",
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        max_iter=2000,  # More iterations for final model
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=42,
    )
    final_model.fit(X_train, y_train)
    
    print(f"Training iterations: {final_model.n_iter_}")
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    train_metrics = evaluate(final_model, X_train, y_train)
    val_metrics = evaluate(final_model, X_val, y_val)
    test_metrics = evaluate(final_model, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Split':<10} | {'Acc':>8} | {'Rec_S':>8} | {'Rec_U':>8} | {'F1':>8} | {'AUC':>8}")
    print("-" * 65)
    print(f"{'Train':<10} | {train_metrics['accuracy']:>8.4f} | {train_metrics['recall_safe']:>8.4f} | "
          f"{train_metrics['recall_unsafe']:>8.4f} | {train_metrics['f1']:>8.4f} | {train_metrics['roc_auc']:>8.4f}")
    print(f"{'Val':<10} | {val_metrics['accuracy']:>8.4f} | {val_metrics['recall_safe']:>8.4f} | "
          f"{val_metrics['recall_unsafe']:>8.4f} | {val_metrics['f1']:>8.4f} | {val_metrics['roc_auc']:>8.4f}")
    print(f"{'Test':<10} | {test_metrics['accuracy']:>8.4f} | {test_metrics['recall_safe']:>8.4f} | "
          f"{test_metrics['recall_unsafe']:>8.4f} | {test_metrics['f1']:>8.4f} | {test_metrics['roc_auc']:>8.4f}")
    
    # Compare with baseline
    print("\n--- Comparison with baseline MLP (64-32) ---")
    baseline = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    baseline.fit(X_train, y_train)
    baseline_test = evaluate(baseline, X_test, y_test)
    
    print(f"{'Baseline':<10} | {baseline_test['accuracy']:>8.4f} | {baseline_test['recall_safe']:>8.4f} | "
          f"{baseline_test['recall_unsafe']:>8.4f} | {baseline_test['f1']:>8.4f} | {baseline_test['roc_auc']:>8.4f}")
    print(f"{'Deep MLP':<10} | {test_metrics['accuracy']:>8.4f} | {test_metrics['recall_safe']:>8.4f} | "
          f"{test_metrics['recall_unsafe']:>8.4f} | {test_metrics['f1']:>8.4f} | {test_metrics['roc_auc']:>8.4f}")
    
    delta_auc = test_metrics['roc_auc'] - baseline_test['roc_auc']
    print(f"\nROC-AUC improvement: {delta_auc:+.4f}")
    
    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACTS_DIR / "deep_mlp_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    
    with (ARTIFACTS_DIR / "best_params.json").open("w") as f:
        json.dump({
            "best_optuna_auc": study.best_value,
            "best_params": best_params,
            "architecture": list(layers),
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_test,
        }, f, indent=2)
    
    with (ARTIFACTS_DIR / "feature_columns.json").open("w") as f:
        json.dump(feature_cols, f)
    
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
