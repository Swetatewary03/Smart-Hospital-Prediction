import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    MODEL_ARTIFACTS,
    NUMERIC_FEATURES,
    TARGET_CLASSES,
    TARGET_COLUMN,
    load_hospital_data,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def build_models() -> dict[str, ClassifierMixin]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest (Bagging)": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def main() -> None:
    data = load_hospital_data()
    if data[TARGET_COLUMN].nunique() != len(TARGET_CLASSES):
        raise ValueError("Training data must contain all three waiting-time classes.")

    features = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = data[TARGET_COLUMN].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    results: list[dict[str, float | str]] = []
    model_scores: dict[str, tuple[float, float]] = {}
    models = build_models()
    for name, estimator in models.items():
        evaluation_pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                ("model", estimator),
            ]
        )
        evaluation_pipeline.fit(X_train, y_train)
        predictions = evaluation_pipeline.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision (macro)": precision_score(
                y_test,
                predictions,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            ),
            "Recall (macro)": recall_score(
                y_test,
                predictions,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            ),
            "F1 (macro)": f1_score(
                y_test,
                predictions,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            ),
        }
        results.append(metrics)
        model_scores[name] = (
            float(metrics["F1 (macro)"]),
            float(metrics["Accuracy"]),
        )
        print(
            f"{name}: accuracy={metrics['Accuracy']:.3f}, "
            f"macro-F1={metrics['F1 (macro)']:.3f}"
        )

    best_model_name = max(model_scores, key=model_scores.__getitem__)

    # Refit each candidate on all valid visits after selecting by held-out macro-F1.
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(features, target)
        fitted_models[name] = pipeline
        joblib.dump(pipeline, BASE_DIR / MODEL_ARTIFACTS[name])

    joblib.dump(fitted_models[best_model_name], BASE_DIR / "best_wait_time_model.pkl")
    (BASE_DIR / "best_model_name.txt").write_text(best_model_name, encoding="utf-8")
    pd.DataFrame(results).sort_values(
        ["F1 (macro)", "Accuracy"], ascending=False
    ).to_csv(BASE_DIR / "model_comparison.csv", index=False)

    print(f"\nSelected model: {best_model_name}")
    print(f"Training visits: {len(features)} (cancelled visits excluded)")
    print(f"Artifacts saved to: {BASE_DIR}")


if __name__ == "__main__":
    main()
