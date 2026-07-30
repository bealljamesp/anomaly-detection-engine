import numpy as np
import shap


def explain_anomalies(
    model, X: np.ndarray, feature_names: list[str]
) -> shap.Explanation:
    """Generates SHAP feature importance values for auditability and governance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values
