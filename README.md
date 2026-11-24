# 🏥 CareNova – Smart Hospital Patient Waiting-Time Prediction System

### 💙 SMARTER WAITING, BETTER CARE.

CareNova is a Smart Hospital Patient Waiting-Time Prediction System designed to improve hospital queue management and enhance the patient experience using Machine Learning.

The system predicts patient waiting-time categories based on hospital department, triage level, visit type, and day of the week. It provides an interactive dashboard for monitoring patient queues, doctor availability, department analytics, and waiting-time predictions.

---

## 🚀 Key Features

- **Machine Learning Prediction:** Predicts patient waiting-time categories using trained classification models.
- **Patient Queue Management:** Monitors patient tokens, queue positions, waiting times, and consultation status.
- **Doctor Availability:** Displays doctor availability, department, consultation status, and assigned consultation rooms.
- **Department Analytics:** Analyzes department-wise patient distribution, workload, and average waiting times.
- **Patient Portal:** Allows patients to view their own appointment information, queue status, notifications, and care journey.
- **Admin Dashboard:** Provides an overview of hospital operations and patient flow.
- **Interactive Visualizations:** Displays hospital statistics, trends, charts, and analytics.
- **Role-Based Access:** Provides separate access for Admin, Staff, and Patients.
- **Light and Dark Mode:** Supports different dashboard viewing preferences.

---

## 🧠 Machine Learning – Waiting-Time Classification

CareNova uses supervised Machine Learning classification algorithms to predict patient waiting-time categories.

### Input Features

The trained classifier uses four features:

1. **Department:** Hospital department handling the patient.
2. **Triage Level:** Patient's assigned triage level.
3. **Visit Type:** Appointment or walk-in.
4. **Visit Day:** Weekday derived from the visit date.

### Classification Algorithms

The training program compares four classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

The best-performing model is selected using the **Macro F1 Score** on a stratified held-out test set.

Cancelled visits are excluded during model training.

### Waiting-Time Categories

| Category | Waiting Time |
|---|---|
| Low | Up to 20 minutes |
| Medium | Over 20 and up to 45 minutes |
| High | Over 45 minutes |

### ML Workflow

Data Collection → Data Preprocessing → Feature Preparation → Model Training → Model Evaluation → Best Model Selection → Waiting-Time Classification

---

## 📊 Dashboard Modules

| Module | Description |
|---|---|
| Admin Dashboard | Hospital-wide operational overview |
| Patient Queue | Patient registration and queue monitoring |
| Waiting-Time ML | Machine Learning-based waiting-time prediction |
| Doctor Availability | Doctor status and room management |
| Department Analytics | Department-wise hospital performance |
| Patient Portal View | Personal appointment and queue information |
| Settings | Dashboard and account preferences |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Chart.js |
| Model Storage | Pickle |
| Version Control | Git and GitHub |

---

## 📁 Repository Structure

```text
Smart-Hospital-Prediction/
│
├── app.py                  # Main Flask application
├── preprocess.py           # Data preprocessing
├── prepare_data.py         # Dataset preparation
├── train_model.py          # ML model training
├── ml_utils.py             # Machine Learning utilities
│
├── dataset/                # Dataset files
├── static/                 # CSS, JavaScript and assets
├── templates/              # HTML templates
├── tests/                  # Backend test files
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Swetatewary03/Smart-Hospital-Prediction.git
```

### 2. Navigate to the Project Directory

```powershell
cd Smart-Hospital-Prediction
```

### 3. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 4. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 6. Train the Machine Learning Models

```powershell
python train_model.py
```

### 7. Start the Flask Application

```powershell
python app.py
```

Open the local address displayed in the terminal to access CareNova.

---

## 🔌 Frontend / API Contract

CareNova provides a Flask API for Machine Learning-based waiting-time classification.

### POST `/api/predict`

Accepts JSON input containing:

| Parameter | Type | Description |
|---|---|---|
| Department | String | Hospital department |
| Triage_Level | Integer | Triage level from 1 to 5 |
| Visit_Type | String | Appointment or Walk-in |
| Visit_Day | String | Day of the week; defaults to Wednesday for older clients |

### Example Request

```json
{
  "Department": "Cardiology",
  "Triage_Level": 3,
  "Visit_Type": "Appointment",
  "Visit_Day": "Wednesday"
}
```

### Response

The API returns each classifier's predicted waiting-time category, along with the selected model and its prediction.

Possible prediction categories:

- Low
- Medium
- High

### GET `/health`

Checks whether the Flask backend has successfully loaded its trained Machine Learning models.

---

## 🧪 Model Evaluation and Testing

CareNova evaluates its classification models using a stratified held-out test set.

**Model Selection Metric:** Macro F1 Score

Macro F1 calculates the F1 score independently for each class and takes their unweighted average.

This helps evaluate model performance across all three waiting-time categories.

### Run Backend Tests

```powershell
python -m unittest discover -s tests
```

---

## 🎯 Project Objectives

- Reduce uncertainty in hospital patient waiting times.
- Improve patient queue management.
- Support efficient doctor and department utilization.
- Provide data-driven insights for hospital administration.
- Improve transparency in the patient waiting experience.
- Support better hospital resource planning.

---

## 🔮 Future Enhancements

- Integration with real-time hospital management systems.
- Improved waiting-time prediction accuracy.
- Advanced hospital resource allocation.
- Mobile-friendly patient access.
- Enhanced analytics and reporting.
- Integration with hospital appointment systems.

---

## 👩‍💻 Project Information

| Field | Details |
|---|---|
| Project Name | CareNova |
| Project Domain | Smart Healthcare |
| Project Category | Machine Learning |
| Project Type | Academic Project |
| Backend | Python and Flask |
| Machine Learning | Scikit-learn |
| Frontend | HTML, CSS and JavaScript |
| Version Control | Git and GitHub |

---

## 📌 Disclaimer

CareNova is an academic project developed for educational and demonstration purposes. Machine Learning predictions are estimates and should not be treated as guaranteed real-world hospital waiting times or as a substitute for clinical triage decisions.

---

### 💙 CareNova
**SMARTER WAITING, BETTER CARE.**