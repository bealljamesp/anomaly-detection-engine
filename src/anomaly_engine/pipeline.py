import numpy as np
import polars as pl


def ingest_cloud_ledger(source_uri: str) -> pl.DataFrame:
    """Ingests transactional ledgers directly from Cloud Object Storage (AWS S3, GCP, or Parquet).

    Args:
        source_uri: Path or cloud URI (e.g., 's3://ledger-bucket/transactions.parquet' or local file).
    """
    return pl.read_parquet(source_uri)


def generate_synthetic_transactions(
    n_samples: int = 10000, anomaly_rate: float = 0.025, seed: int = 42
) -> pl.DataFrame:
    """Generates synthetic transactional dataset with injected anomalies."""
    np.random.seed(seed)

    amount = np.random.exponential(scale=100.0, size=n_samples)
    transaction_count = np.random.poisson(lam=5, size=n_samples)
    risk_score_raw = np.random.normal(loc=50, scale=10, size=n_samples)

    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_idx = np.random.choice(n_samples, size=n_anomalies, replace=False)

    amount[anomaly_idx] *= np.random.uniform(5, 15, size=n_anomalies)
    transaction_count[anomaly_idx] += np.random.randint(20, 50, size=n_anomalies)
    risk_score_raw[anomaly_idx] += np.random.uniform(30, 50, size=n_anomalies)

    return pl.DataFrame(
        {
            "transaction_id": [f"TXN-{i:06d}" for i in range(n_samples)],
            "amount": amount,
            "daily_txn_count": transaction_count,
            "raw_risk_score": risk_score_raw,
        }
    )


def preprocess_features(df: pl.DataFrame) -> np.ndarray:
    """Extracts and normalizes features using vectorized Polars operations."""
    feature_df = df.select(
        [
            pl.col("amount").log1p(),
            pl.col("daily_txn_count"),
            pl.col("raw_risk_score"),
        ]
    )
    return feature_df.to_numpy()


# Pandas implementation for preprocessing_features:

# import pandas as pd


# def preprocess_features(df: pd.DataFrame) -> np.ndarray:
#   """Extracts and normalizes features using vectorized Pandas operations."""
#   feature_df = pd.DataFrame({
#       "amount": np.log1p(df["amount"]),
#       "daily_txn_count": df["daily_txn_count"],
#       "raw_risk_score": df["raw_risk_score"],
#   })
#   return feature_df.to_numpy()
