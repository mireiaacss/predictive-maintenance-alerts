"""Retrain the binary failure model used by the dashboard.

Reproduces the final Random Forest from notebooks/predictive_maintenance.ipynb
(best hyperparameters found with GridSearchCV) and saves it to models/failure_model.pkl.

Usage:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "predictive_maintenance.csv"
MODEL_PATH = ROOT / "models" / "failure_model.pkl"

NUMERIC_FEATURES = [
    "Air temperature [°C]",
    "Process temperature [°C]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Kelvin -> Celsius
    for col in ["Air temperature [K]", "Process temperature [K]"]:
        df[col.replace("[K]", "[°C]")] = df[col] - 273.15
    df = df.drop(columns=["Air temperature [K]", "Process temperature [K]"])

    # Drop inconsistent labels (Target and Failure Type disagree)
    inconsistent = ((df["Failure Type"] == "No Failure") & (df["Target"] == 1)) | (
        (df["Failure Type"] != "No Failure") & (df["Target"] == 0)
    )
    return df[~inconsistent]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 5},
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
