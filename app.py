
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACTS = {
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest (Bagging)": "hospital_waiting_time_model.pkl",
}


def load_models() -> tuple[dict[str, Pipeline], str]:
    model_path = BASE_DIR / "best_wait_time_model.pkl"
    name_path = BASE_DIR / "best_model_name.txt"
    required_paths = [model_path, name_path]
    required_paths.extend(BASE_DIR / filename for filename in MODEL_ARTIFACTS.values())
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(
            "Trained model files are missing: "
            + ", ".join(missing_paths)
            + ". Run train_model.py first."
        )

    models: dict[str, Pipeline] = {
        name: joblib.load(BASE_DIR / filename)
        for name, filename in MODEL_ARTIFACTS.items()
    }
    best_model_name = name_path.read_text(encoding="utf-8").strip()
    if best_model_name not in models:
        raise RuntimeError(
            f"Unknown best model '{best_model_name}'. Run train_model.py again."
        )
    models[best_model_name] = joblib.load(model_path)
    return models, best_model_name


models, best_model_name = load_models()


def validate_patient_data(
    payload: object,
) -> tuple[pd.DataFrame | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, "Request body must be an object."

    department = payload.get("Department")
    visit_type = payload.get("Visit_Type")
    visit_day = payload.get("Visit_Day", "Wednesday")
    triage = payload.get("Triage_Level")
    if not isinstance(department, str) or not department.strip():
        return None, "Department is required."
    if not isinstance(visit_type, str) or not visit_type.strip():
        return None, "Visit_Type is required."
    valid_days = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    }
    if not isinstance(visit_day, str) or visit_day not in valid_days:
        return None, "Visit_Day must be a weekday name."
    if isinstance(triage, bool):
        return None, "Triage_Level must be an integer from 1 to 5."
    try:
        triage_number = float(triage)
    except (TypeError, ValueError):
        return None, "Triage_Level must be an integer from 1 to 5."
    if not triage_number.is_integer() or not 1 <= triage_number <= 5:
        return None, "Triage_Level must be an integer from 1 to 5."

    row = pd.DataFrame(
        [
            {
                "Department": department.strip(),
                "Triage_Level": int(triage_number),
                "Visit_Type": visit_type.strip(),
                "Visit_Day": visit_day,
            }
        ]
    )
    return row, None


def get_predictions(patient_data: pd.DataFrame) -> dict[str, str]:
    return {
        name: str(model.predict(patient_data)[0])
        for name, model in models.items()
    }


# Homepage
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/dashboard")
def dashboard():
    return render_template("Dashboard.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    patient_data, error = validate_patient_data(request.form)
    if patient_data is None:
        return render_template(
            "predict.html", error=error or "Unable to read patient details."
        ), 400
    predictions = get_predictions(patient_data)
    return render_template(
        "predict.html",
        predictions=predictions,
        best_prediction=predictions[best_model_name],
        best_model_name=best_model_name,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify(error="Content-Type must be application/json."), 415
    patient_data, error = validate_patient_data(request.get_json(silent=True))
    if patient_data is None:
        return jsonify(error=error or "Unable to read patient details."), 400
    predictions = get_predictions(patient_data)
    return jsonify(
        predictions=predictions,
        best_prediction=predictions[best_model_name],
        best_model_name=best_model_name,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", best_model_name=best_model_name)


# Run Flask application
if __name__ == '__main__':
    app.run(debug=False)
