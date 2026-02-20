"""
Multi-model comparison for ai-v0 safe classification.
Compares: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, MLP
"""
import json
import warnings
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).with_name("proto3-hybrid_samples.csv")
RESULTS_PATH = Path(__file__).with_name("model_comparison_results.json")


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
    
    return X_train_s, X_val_s, X_test_s, y_train.values, y_val.values, y_test.values


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_safe": recall_score(y_true, y_pred, pos_label=1),
        "recall_unsafe": recall_score(y_true, y_pred, pos_label=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
    }


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", 
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1.0, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss"
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1
        ),
        "MLP (64-32)": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=500, random_state=42, early_stopping=True
        ),
        "MLP (128-64-32)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            max_iter=500, random_state=42, early_stopping=True
        ),
    }


def main():
    print("=" * 80)
    print("Multi-Model Comparison for Safe/Unsafe Classification")
    print("=" * 80)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"\n[{name}] Training...", end=" ", flush=True)
        
        start = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["train_time_sec"] = round(train_time, 2)
        
        results[name] = metrics
        
        print(f"Done ({train_time:.1f}s)")
        auc_display = f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else "N/A"
        print(f"  Acc: {metrics['accuracy']:.3f} | "
              f"Rec_safe: {metrics['recall_safe']:.3f} | "
              f"Rec_unsafe: {metrics['recall_unsafe']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"AUC: {auc_display}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (Test Set)")
    print("=" * 80)
    
    print(f"\n{'Model':<22} | {'Acc':>6} | {'Rec_S':>6} | {'Rec_U':>6} | {'F1':>6} | {'AUC':>6} | {'Time':>6}")
    print("-" * 80)
    
    # Sort by ROC-AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]["roc_auc"] or 0, reverse=True)
    
    for name, m in sorted_results:
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "N/A"
        print(f"{name:<22} | {m['accuracy']:>6.4f} | {m['recall_safe']:>6.4f} | "
              f"{m['recall_unsafe']:>6.4f} | {m['f1']:>6.4f} | {auc_str:>6} | {m['train_time_sec']:>5.1f}s")
    
    # Best model analysis
    print("\n" + "=" * 80)
    print("BEST MODELS BY METRIC")
    print("=" * 80)
    
    best_auc = max(results.items(), key=lambda x: x[1]["roc_auc"] or 0)
    best_unsafe = max(results.items(), key=lambda x: x[1]["recall_unsafe"])
    best_safe = max(results.items(), key=lambda x: x[1]["recall_safe"])
    best_f1 = max(results.items(), key=lambda x: x[1]["f1"])
    
    print(f"Best ROC-AUC:      {best_auc[0]} ({best_auc[1]['roc_auc']:.4f})")
    print(f"Best Unsafe Recall: {best_unsafe[0]} ({best_unsafe[1]['recall_unsafe']:.4f})")
    print(f"Best Safe Recall:   {best_safe[0]} ({best_safe[1]['recall_safe']:.4f})")
    print(f"Best F1 Score:      {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    
    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
