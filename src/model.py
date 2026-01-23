import joblib
from src.config import MODEL_FILE


class ComplaintPriorityModel:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_FILE)

    def predict_proba(self, text: str) -> float:
        """
        Returns probability that a complaint is high priority.
        """
        proba = self.pipeline.predict_proba([text])[0][1]
        return float(proba)