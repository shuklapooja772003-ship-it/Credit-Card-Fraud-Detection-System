import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
FRAUD_THRESHOLD = 0.50


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv and place it in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    if df.empty or df["Class"].isna().any():
        raise ValueError("Dataset is empty or target labels contain missing values.")

    df["Class"] = pd.to_numeric(df["Class"], errors="raise")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("'Class' must contain both 0 (legitimate) and 1 (fraud).")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Duplicate rows (reported, not automatically removed): {df.duplicated().sum()}")
    print("\nClass distribution:\n", df["Class"].value_counts())

    X = df.drop(columns=["Class"]).apply(pd.to_numeric, errors="raise")
    y = df["Class"].astype(int)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values remain.")
    return X, y


def evaluate_model(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= FRAUD_THRESHOLD).astype(int)
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities),
    }
    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    print(classification_report(
        y_test, predictions, target_names=["Legitimate", "Fraud"], zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    return results, probabilities, predictions


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    results, fitted, probabilities_by_model, predictions_by_model = [], {}, {}, {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        result, probs, preds = evaluate_model(name, model, X_test, y_test)
        results.append(result)
        fitted[name] = model
        probabilities_by_model[name] = probs
        predictions_by_model[name] = preds

    results_df = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "fraud_model.pkl")

    metadata = {
        "model_name": best_name,
        "features": list(X.columns),
        "threshold": FRAUD_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "primary_selection_metric": "PR_AUC",
        "threshold_note": "Default demonstration threshold; tune on validation data for intended operating costs."
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=4), encoding="utf-8"
    )

    best_preds = predictions_by_model[best_name]
    best_probs = probabilities_by_model[best_name]
    report = classification_report(
        y_test, best_preds, target_names=["Legitimate", "Fraud"], zero_division=0
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\nThreshold: {FRAUD_THRESHOLD}\n\n{report}\n\n"
        f"ROC-AUC: {roc_auc_score(y_test, best_probs):.5f}\n"
        f"PR-AUC: {average_precision_score(y_test, best_probs):.5f}\n",
        encoding="utf-8"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, best_preds),
        display_labels=["Legitimate", "Fraud"]
    )
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nModel comparison:\n", results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'fraud_model.pkl'}")
    print(f"Saved reports in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
