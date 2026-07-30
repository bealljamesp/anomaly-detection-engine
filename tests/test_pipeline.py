import polars as pl

from anomaly_engine.model import AnomalyDetector
from anomaly_engine.pipeline import (
    generate_synthetic_transactions,
    ingest_cloud_ledger,
    preprocess_features,
)


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
    assert -1 in preds


def test_parquet_roundtrip(tmp_path):
    """Tests cloud-native Parquet storage and ingestion pipeline."""
    file_path = tmp_path / "test_ledger.parquet"
    df_orig = generate_synthetic_transactions(n_samples=50)
    df_orig.write_parquet(file_path)

    df_ingested = ingest_cloud_ledger(str(file_path))
    assert df_ingested.height == 50
    assert "transaction_id" in df_ingested.columns


# def test_model_contamination_failure():
#     """Deliberately failing test to observe pytest error introspection."""
#     df = generate_synthetic_transactions(n_samples=1000, anomaly_rate=0.025)
#     X = preprocess_features(df)

#     detector = AnomalyDetector(contamination=0.025)
#     preds, _ = detector.fit_predict(X)

#     n_anomalies_detected = (preds == -1).sum()

#     # We expect 25 anomalies (1000 * 0.025), but we assert 999 to force a failure
#     assert n_anomalies_detected == 999
