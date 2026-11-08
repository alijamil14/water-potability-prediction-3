from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "water_potability_model.pkl"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"

def main():
    df = pd.read_csv(DATA_PATH)

    # Validate the expected dataset structure before training.
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Keep the test set separate. The imputer is fitted only on training data
    # because it is inside the pipeline, preventing data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "feature_names": FEATURES,
        "target_names": {"0": "Not potable", "1": "Potable"},
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Evaluation metrics saved to: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1_score']:.3f}")
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
