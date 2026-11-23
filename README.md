# Smart Hospital Waiting-Time Classifier

## Backend setup

From the project directory, create and activate a virtual environment, then install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train the classifiers before starting the Flask app:

```powershell
python train_model.py
python app.py
```

The training program excludes cancelled visits, compares logistic regression,
K-nearest neighbors, decision tree, and random forest classifiers on a stratified
held-out set, and selects the model by macro F1 score. The model uses Department, Triage_Level, Visit_Type, and the weekday derived from
Visit_Date, all available at prediction time.
All four algorithms are covered by the provided course syllabus; no HMM or
clustering methods are used.

Waiting-time categories are `Low` (up to 20 minutes), `Medium` (over 20 and up
to 45 minutes), and `High` (over 45 minutes).

Run the backend tests with:

```powershell
python -m unittest discover -s tests
```

## Frontend/API contract

`POST /api/predict` accepts JSON with `Department`, `Visit_Type`, and optional
`Visit_Day` strings, plus an integer `Triage_Level` from 1 to 5. `Visit_Day`
defaults to `Wednesday` for older clients:

```json
{
  "Department": "Cardiology",
  "Triage_Level": 3,
  "Visit_Type": "Appointment",
  "Visit_Day": "Wednesday"
}
```

It returns each model's predicted `Low`, `Medium`, or `High` waiting-time
category, together with the selected model and its prediction. `GET /health`
reports whether the backend has loaded its trained models.
