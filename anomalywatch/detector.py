import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ALGORITHMS = {
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
}


def detect_anomalies(
    data: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
    algorithm: str = "Isolation Forest",
) -> pd.DataFrame:
    """
    Detect anomalies in numerical tabular data.

    Returns:
        Original DataFrame with:
        - anomaly_score: larger = more anomalous
        - is_anomaly: boolean anomaly prediction
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if not features:
        raise ValueError("Select at least one numerical feature.")

    if not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be in (0, 0.5].")

    if len(data) < 10:
        raise ValueError("At least 10 observations are required.")

    if len(set(features)) != len(features):
        raise ValueError("Duplicate features are not allowed.")

    if any(feature not in data.columns for feature in features):
        raise ValueError("A selected feature is missing.")

    X = data[features].copy()

    if not all(
        pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
        for col in X.columns
    ):
        raise ValueError("Features must be numerical, not boolean.")

    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(X.median())

    if X.isna().any().any():
        raise ValueError("Features cannot be entirely missing.")

    X_scaled = StandardScaler().fit_transform(X)

    if algorithm == "Isolation Forest":
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        predictions = model.fit_predict(X_scaled)
        scores = -model.score_samples(X_scaled)

    elif algorithm == "Local Outlier Factor":
        model = LocalOutlierFactor(
            n_neighbors=min(20, len(X_scaled) - 1),
            contamination=contamination,
        )
        predictions = model.fit_predict(X_scaled)
        scores = -model.negative_outlier_factor_

    else:
        # nu controls an upper bound on the training
        # fraction of errors and a lower bound on
        # the fraction of support vectors.
        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=contamination,
        )
        predictions = model.fit_predict(X_scaled)
        scores = -model.decision_function(X_scaled).ravel()

    result = data.copy()

    result["anomaly_score"] = scores
    result["is_anomaly"] = predictions == -1

    return result
