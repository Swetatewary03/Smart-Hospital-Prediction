import unittest

from app import app, best_model_name


class PredictionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self.patient = {
            "Department": "Cardiology",
            "Triage_Level": 3,
            "Visit_Type": "Appointment",
            "Visit_Day": "Wednesday",
        }

    def test_api_returns_all_wait_categories(self) -> None:
        response = self.client.post("/api/predict", json=self.patient)

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["best_model_name"], best_model_name)
        self.assertEqual(result["best_prediction"], result["predictions"][best_model_name])
        self.assertEqual(len(result["predictions"]), 4)
        self.assertTrue(
            set(result["predictions"].values()).issubset({"Low", "Medium", "High"})
        )

    def test_api_rejects_triage_outside_supported_range(self) -> None:
        invalid_patient = {**self.patient, "Triage_Level": 6}
        response = self.client.post("/api/predict", json=invalid_patient)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Triage_Level", response.get_json()["error"])

    def test_api_rejects_invalid_visit_day(self) -> None:
        invalid_patient = {**self.patient, "Visit_Day": "Funday"}
        response = self.client.post("/api/predict", json=invalid_patient)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Visit_Day", response.get_json()["error"])

    def test_api_defaults_missing_visit_day(self) -> None:
        patient_without_day = {
            key: value for key, value in self.patient.items() if key != "Visit_Day"
        }
        response = self.client.post("/api/predict", json=patient_without_day)

        self.assertEqual(response.status_code, 200)

    def test_api_rejects_malformed_or_missing_json_fields(self) -> None:
        response = self.client.post("/api/predict", json={"Department": "Cardiology"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Visit_Type", response.get_json()["error"])

    def test_form_prediction_still_works(self) -> None:
        response = self.client.post("/predict", data=self.patient)

        self.assertEqual(response.status_code, 200)
        self.assertIn(best_model_name.encode(), response.data)


if __name__ == "__main__":
    unittest.main()
