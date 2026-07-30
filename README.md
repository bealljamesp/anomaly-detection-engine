# Cloud-Native Anomaly Engine & MLOps Governance Framework

[![CI Test Suite](https://github.com/bealljamesp/anomaly-detection-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/bealljamesp/anomaly-detection-engine/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview

The **Anomaly Detection Engine** is a modular, high-throughput Python framework designed to ingest, process, and evaluate high-dimensional transactional ledgers for latent anomalies, system logic failures, and compliance drift.

Built with an emphasis on **rigorous MLOps governance**, this project demonstrates end-to-end data pipeline vectorization using **Polars**, unsupervised machine learning via **Isolation Forests**, automated unit testing with **Pytest**, continuous integration through **GitHub Actions**, and model interpretability using **SHAP (SHapley Additive exPlanations)**.

This architecture reflects production oversight analytics applied in high-stakes auditing, risk scoring, and federal data governance environments (e.g., DHS Office of Inspector General).

---

## 🛠️ Key Features & Technical Stack

* **High-Throughput Vectorized Ingestion:** Leverages `Polars` for $O(1)$ memory allocation profiles and ultra-fast DataFrame transformations over transactional datasets.
* **Unsupervised Anomaly Detection:** Implements an `IsolationForest` model to detect multivariate outliers, risk score transactional patterns, and flag statistical anomalies.
* **Model Explainability & Auditability:** Integrates SHAP game-theoretic feature attribution to decompose anomaly scores into transparent, human-readable explanations.
* **Production MLOps Governance:**
  * Automated testing suite using `pytest` for pipeline validation and contamination assertions.
  * Continuous Integration (CI) pipeline powered by `GitHub Actions` running Python 3.12.
  * Modern Python packaging structured around `pyproject.toml` standards.

---

## 🏗️ Repository Architecture

```text
anomaly-detection-engine/
├── .github/
│   └── workflows/
│       └── tests.yml           # Automated CI/CD pipeline running pytest on Python 3.12
├── src/
│   └── anomaly_engine/
│       ├── __init__.py         # Package entry point
│       ├── pipeline.py        # Polars-based data generation & feature preprocessing
│       ├── model.py           # Isolation Forest anomaly detection engine
│       └── explainability.py  # SHAP game-theoretic model interpretability module
├── tests/
│   └── test_pipeline.py       # Pytest suite for unit & integration testing
├── main.py                    # Pipeline execution script
├── pyproject.toml             # Package dependencies and dev configurations
└── README.md                  # Technical documentation
