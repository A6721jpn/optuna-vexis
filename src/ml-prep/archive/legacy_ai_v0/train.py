import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("artifacts")


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


def train_model(x_train, y_train, seed: int) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x, y):
    probas = model.predict_proba(x)[:, 1]
    preds = (probas >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probas),
        "classification_report": classification_report(y, preds, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v0 safe classifier.")
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
    data = load_dataset(args.csv)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        data, args.label, args.seed
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    model = train_model(x_train_scaled, y_train, args.seed)

    metrics = {
        "train": evaluate_model(model, x_train_scaled, y_train),
        "val": evaluate_model(model, x_val_scaled, y_val),
        "test": evaluate_model(model, x_test_scaled, y_test),
    }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifacts / "mlp_model.joblib")
    joblib.dump(scaler, args.artifacts / "scaler.joblib")

    with (args.artifacts / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (args.artifacts / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(list(x_train.columns), f, indent=2, ensure_ascii=False)

    print("Training complete. Metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    main()
