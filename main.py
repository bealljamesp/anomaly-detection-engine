import polars as pl

from anomaly_engine.model import AnomalyDetector
from anomaly_engine.pipeline import (
    generate_synthetic_transactions,
    preprocess_features,
)


def main():
    RATE = 0.025

    print("🚀 Running Anomaly Detection Pipeline...")

    # 1. Ingest Data with 2.5% injected anomalies
    df = generate_synthetic_transactions(n_samples=5000, anomaly_rate=RATE)
    print(f"Ingested {df.height} records successfully.")

    # 2. Preprocess Features
    X = preprocess_features(df)

    # 3. Fit Detector with 2.5% contamination assumption
    detector = AnomalyDetector(contamination=RATE)
    preds, scores = detector.fit_predict(X)

    # 4. Attach Results
    df = df.with_columns(
        [
            pl.Series("anomaly_flag", preds == -1),
            pl.Series("anomaly_score", scores),
        ]
    )

    anomalies = df.filter(pl.col("anomaly_flag"))
    print(f"⚠️ Detected {anomalies.height} anomalous transactions.")
    print(anomalies.head(5))

    # Summary Mean Statistics
    normal_stats = df.filter(~pl.col("anomaly_flag")).select(
        [
            pl.col("amount").mean(),
            pl.col("daily_txn_count").mean(),
            pl.col("raw_risk_score").mean(),
        ]
    )

    anomaly_stats = df.filter(pl.col("anomaly_flag")).select(
        [
            pl.col("amount").mean(),
            pl.col("daily_txn_count").mean(),
            pl.col("raw_risk_score").mean(),
        ]
    )

    print("\n--- BASELINE (NORMAL) TRANSACTIONS ---")
    print(normal_stats)

    print("\n--- FLAGGED ANOMALOUS TRANSACTIONS ---")
    print(anomaly_stats)


if __name__ == "__main__":
    main()
