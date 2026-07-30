import polars as pl

from anomaly_engine.model import AnomalyDetector
from anomaly_engine.pipeline import generate_synthetic_transactions, preprocess_features


def test_data_generation():
    df = generate_synthetic_transactions(n_samples=100)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 100


def test_anomaly_detection():
    df = generate_synthetic_transactions(n_samples=200, anomaly_rate=0.025)
    X = preprocess_features(df)

    detector = AnomalyDetector(contamination=0.025)
    preds, scores = detector.fit_predict(X)

    assert len(preds) == 200
    assert -1 in preds  # Ensure anomalies were detected


# def test_model_contamination_failure():
#     """Deliberately failing test to observe pytest error introspection."""
#     df = generate_synthetic_transactions(n_samples=1000, anomaly_rate=0.025)
#     X = preprocess_features(df)

#     detector = AnomalyDetector(contamination=0.025)
#     preds, _ = detector.fit_predict(X)

#     n_anomalies_detected = (preds == -1).sum()

#     # We expect 25 anomalies (1000 * 0.025), but we assert 999 to force a failure
#     assert n_anomalies_detected == 999
