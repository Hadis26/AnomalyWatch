
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from anomalywatch.detector import detect_anomalies


def main():
    data = pd.read_csv("data/sample_sensors.csv")

    features = [
        "temperature_C",
        "vibration",
        "pressure_kPa",
    ]

    result = detect_anomalies(
        data,
        features=features,
        contamination=0.04,
    )

    y_true = data["injected_anomaly"].astype(bool)
    y_pred = result["is_anomaly"].astype(bool)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()

    metrics = {
        "observations": int(len(data)),
        "true_anomalies": int(y_true.sum()),
        "detected_anomalies": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    output = Path("results")
    output.mkdir(exist_ok=True)

    (output / "evaluation.json").write_text(
        json.dumps(metrics, indent=2)
    )

    result.to_csv(
        output / "labeled_sensors.csv",
        index=False,
    )

    print("\nAnomalyWatch Evaluation")
    print("-----------------------")

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
