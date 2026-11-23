from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "hospital_visits.csv"
FEATURE_COLUMNS = ["Department", "Triage_Level", "Visit_Type", "Visit_Day"]
CATEGORICAL_FEATURES = ["Department", "Visit_Type", "Visit_Day"]
NUMERIC_FEATURES = ["Triage_Level"]
TARGET_COLUMN = "Wait_Category"
TARGET_CLASSES = ["Low", "Medium", "High"]
MODEL_ARTIFACTS = {
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest (Bagging)": "hospital_waiting_time_model.pkl",
}


def load_hospital_data() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    required_columns = [
        "Department",
        "Triage_Level",
        "Visit_Type",
        "Visit_Date",
        "Waiting_Time",
        "Cancellation_Flag",
    ]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    data = data.loc[data["Cancellation_Flag"].eq(0)].copy()
    data["Visit_Date"] = pd.to_datetime(data["Visit_Date"], errors="coerce")
    data["Visit_Day"] = data["Visit_Date"].dt.day_name()
    data["Triage_Level"] = pd.to_numeric(data["Triage_Level"], errors="coerce")
    data["Waiting_Time"] = pd.to_numeric(data["Waiting_Time"], errors="coerce")
    data = data.dropna(subset=[*FEATURE_COLUMNS, "Waiting_Time", "Cancellation_Flag"])
    data = data.loc[
        data["Triage_Level"].between(1, 5) & data["Waiting_Time"].ge(0)
    ].copy()
    data[TARGET_COLUMN] = pd.cut(
        data["Waiting_Time"],
        bins=[-1, 20, 45, float("inf")],
        labels=TARGET_CLASSES,
    )
    data = data.dropna(subset=[TARGET_COLUMN])
    if data.empty:
        raise ValueError("No valid, non-cancelled hospital visits were found.")
    return data
