"""
ai-v0 Optuna hyperparameter tuning script.
Optimizes MLP architecture and training parameters.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

DEFAULT_DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts_optuna")


def load_dataset(csv_path: Path):
    return pd.read_csv(csv_path)


def create_objective(x_train, y_train, x_val, y_val, seed: int):
    """Create Optuna objective function."""
    
    def objective(trial: optuna.Trial) -> float:
        # Hidden layer architecture
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            n_units = trial.suggest_int(f"n_units_l{i}", 32, 256)
            hidden_layers.append(n_units)
        
        # Training parameters
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        
        # Activation function
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )
        
        model.fit(x_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(x_val)
        y_proba = model.predict_proba(x_val)[:, 1]
        
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        # Optimize for F1 + ROC-AUC combination
        return (f1 + roc_auc) / 2
    
    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for ai-v0")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.artifacts.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.data}")
    data = load_dataset(args.data)
    
    X = data.drop(columns=["safe"]).values
    y = data["safe"].values
    
    # Split data
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.176, random_state=args.seed, stratify=y_train_val
    )
    
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    
    # Create study
    study = optuna.create_study(direction="maximize", study_name="ai-v0-mlp")
    objective = create_objective(x_train_scaled, y_train, x_val_scaled, y_val, args.seed)
    
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    print("\n=== Best Trial ===")
    print(f"Value: {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")
    
    # Train final model with best params
    best_params = study.best_trial.params
    n_layers = best_params["n_layers"]
    hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
    
    final_model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        batch_size=best_params["batch_size"],
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    
    # Train on train+val
    x_train_full = np.vstack([x_train_scaled, x_val_scaled])
    y_train_full = np.hstack([y_train, y_val])
    final_model.fit(x_train_full, y_train_full)
    
    # Evaluate on test
    y_pred = final_model.predict(x_test_scaled)
    y_proba = final_model.predict_proba(x_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_params": best_params,
    }
    
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save artifacts
    joblib.dump(final_model, args.artifacts / "model.joblib")
    joblib.dump(scaler, args.artifacts / "scaler.joblib")
    
    with open(args.artifacts / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(args.artifacts / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nArtifacts saved to {args.artifacts}")


if __name__ == "__main__":
    main()
