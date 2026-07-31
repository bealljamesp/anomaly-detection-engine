import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination: float, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state, n_jobs=-1
        )

    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits Isolation Forest model and predicts anomalies.
        Returns:
            predictions: -1 for anomalies, 1 for normal data.
            scores: Decision function anomaly scores (lower means more anomalous).
        """
        self.model.fit(X)
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        return predictions, scores
