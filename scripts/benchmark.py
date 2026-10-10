from pathlib import Path
import json
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from anomalywatch.detector import detect_anomalies


ALGORITHMS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
]

FEATURES = [
    "temperature_C",
    "vibration",
    "pressure_kPa",
]


def run_benchmark():
    data = pd.read_csv("data/sample_sensors.csv")
    labels = data["injected_anomaly"].astype(bool)

    results = []

    for algorithm in ALGORITHMS:
        start = time.perf_counter()

        predictions = detect_anomalies(
            data,
            features=FEATURES,
            contamination=0.04,
            algorithm=algorithm,
        )

        elapsed = time.perf_counter() - start
        detected = predictions["is_anomaly"].astype(bool)

        metrics = {
            "algorithm": algorithm,
            "precision": precision_score(labels, detected, zero_division=0),
            "recall": recall_score(labels, detected, zero_division=0),
            "f1_score": f1_score(labels, detected, zero_division=0),
            "detected_anomalies": int(detected.sum()),
            "runtime_ms": elapsed * 1000,
        }

        results.append(metrics)

    results_df = pd.DataFrame(results)

    output = Path("results")
    output.mkdir(exist_ok=True)

    results_df.to_csv(
        output / "algorithm_comparison.csv",
        index=False,
    )

    (output / "algorithm_comparison.json").write_text(json.dumps(results, indent=2))

    print("\nAnomalyWatch — Algorithm Comparison")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
