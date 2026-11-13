"""
train_model.py

Software Defect Prediction on the NASA JM1 dataset.

Trains a Random Forest classifier to predict whether a software module is
defective, using static code metrics (lines of code, cyclomatic complexity,
Halstead metrics, etc.) as features.

Pipeline (leakage-safe):
    1. Load data
    2. Train/test split FIRST
    3. SMOTE oversampling fit only on the training fold
    4. Feature scaling fit only on the training fold
    5. Hyperparameter tuning via GridSearchCV (5-fold CV, F1 scoring)
    6. Evaluation on the untouched test fold
    7. Save the trained model to disk

Usage:
    python train_model.py --data data/jm1.csv --output model.pkl
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "defects"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}


def load_data(path: str) -> pd.DataFrame:
    """Load the JM1 dataset from a CSV file."""
    data = pd.read_csv(path)
    print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    missing = data.isnull().sum().sum()
    if missing:
        print(f"Warning: {missing} missing values found; dropping affected rows.")
        data = data.dropna()

    return data


def prepare_target(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the target column to 0/1 integers if it's boolean/string."""
    if data[TARGET_COLUMN].dtype == bool or data[TARGET_COLUMN].dtype == object:
        data[TARGET_COLUMN] = (
            data[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )
    return data


def train(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split, resample, scale, and tune a Random Forest classifier."""
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    print(f"Class distribution before split:\n{y.value_counts()}")

    # Split BEFORE any resampling or scaling to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Resample only the training fold.
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"Training class distribution after SMOTE:\n"
        f"{pd.Series(y_train_res).value_counts()}"
    )

    # Fit the scaler on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning via 5-fold cross-validation.
    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        rf, PARAM_GRID, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train_res)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    return best_model, scaler, X_test_scaled, y_test, X.columns


def evaluate(model, X_test_scaled, y_test) -> None:
    """Print evaluation metrics on the held-out test set."""
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Evaluation on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


def print_feature_importance(model, feature_names) -> None:
    """Print features ranked by importance."""
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importance ---")
    for name, score in ranked:
        print(f"{name:20s} {score:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/jm1.csv", help="Path to the JM1 CSV dataset"
    )
    parser.add_argument(
        "--output", default="model.pkl", help="Path to save the trained model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    data = load_data(args.data)
    data = prepare_target(data)

    model, scaler, X_test_scaled, y_test, feature_names = train(
        data, test_size=args.test_size, random_state=args.random_state
    )

    evaluate(model, X_test_scaled, y_test)
    print_feature_importance(model, feature_names)

    joblib.dump({"model": model, "scaler": scaler}, args.output)
    print(f"\nModel and scaler saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
