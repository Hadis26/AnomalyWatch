
import numpy as np
import pandas as pd
import pytest

from anomalywatch.detector import detect_anomalies



@pytest.mark.parametrize(
    "algorithm",
    [
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
    ],
)
def test_all_algorithms(algorithm, sample_data):
    result = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
        contamination=0.05,
        algorithm=algorithm,
    )

    assert len(result) == len(sample_data)
    assert result["is_anomaly"].dtype == bool
    assert np.isfinite(result["anomaly_score"]).all()
    assert bool(result["is_anomaly"].iloc[-1])


def test_invalid_algorithm(sample_data):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        detect_anomalies(
            sample_data,
            ["temperature"],
            algorithm="Unknown Model",
        )


def test_boolean_feature_rejected(sample_data):
    sample_data["label"] = False

    with pytest.raises(ValueError, match="numerical"):
        detect_anomalies(
            sample_data,
            ["label"],
        )


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 2))
    data[-1] = [20.0, 20.0]

    return pd.DataFrame(
        data,
        columns=["temperature", "vibration"],
    )


def test_detects_extreme_outlier(sample_data):
    result = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
        contamination=0.05,
    )

    assert bool(result["is_anomaly"].iloc[-1])


def test_preserves_original_data(sample_data):
    original = sample_data.copy(deep=True)

    result = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
    )

    pd.testing.assert_frame_equal(sample_data, original)
    assert len(result) == 100
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns


def test_missing_values(sample_data):
    sample_data.loc[3, "temperature"] = np.nan

    result = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
    )

    assert len(result) == 100
    assert np.isfinite(result["anomaly_score"]).all()


def test_invalid_contamination(sample_data):
    with pytest.raises(ValueError):
        detect_anomalies(
            sample_data,
            ["temperature"],
            contamination=0.8,
        )


def test_reproducibility(sample_data):
    a = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
    )

    b = detect_anomalies(
        sample_data,
        ["temperature", "vibration"],
    )

    np.testing.assert_array_equal(
        a["is_anomaly"],
        b["is_anomaly"],
    )
