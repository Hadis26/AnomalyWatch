import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    data: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.

    Returns the original dataframe with:
        anomaly_score: higher means more anomalous
        is_anomaly: boolean anomaly prediction
    """
    if not features:
        raise ValueError("Select at least one numerical feature.")

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be between 0 and 0.5."
        )

    if len(data) < 10:
        raise ValueError("At least 10 observations are required.")

    X = data[features].copy()

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("All selected features must be numerical.")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation handles missing numeric observations.
    X = X.fillna(X.median())

    if X.isna().any().any():
        raise ValueError(
            "Selected features cannot be entirely missing."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    predictions = model.fit_predict(X_scaled)

    # scikit-learn: lower score_samples = more abnormal
    scores = -model.score_samples(X_scaled)

    result = data.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = predictions == -1

    return result