import polars as pl

from anomaly_engine.model import AnomalyDetector
from anomaly_engine.pipeline import generate_synthetic_transactions, preprocess_features


def main():
    RATE = 0.025

    # 1. Ingest Data with 2.5% injected anomalies
    df = generate_synthetic_transactions(n_samples=5000, anomaly_rate=RATE)

    # 2. Preprocess
    X = preprocess_features(df)

    # 3. Fit Detector with 2.5% contamination assumption
    detector = AnomalyDetector(contamination=RATE)
    preds, scores = detector.fit_predict(X)

    # 4. Results
    df = df.with_columns(
        [pl.Series("anomaly_flag", preds == -1), pl.Series("anomaly_score", scores)]
    )

    anomalies = df.filter(pl.col("anomaly_flag"))
    print(f"⚠️ Detected {anomalies.height} anomalous transactions.")


if __name__ == "__main__":
    main()
