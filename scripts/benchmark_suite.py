from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from anomalywatch.detector import detect_anomalies


ALGORITHMS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
]

DATASETS = {
    "Point anomalies": "point_anomalies.csv",
    "Gradual drift": "gradual_drift.csv",
    "Contextual anomalies": "contextual_anomalies.csv",
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmarks"
OUTPUT_DIR = ROOT / "results" / "benchmarks"


def evaluate_dataset(name, filename):
    data = pd.read_csv(DATA_DIR / filename)

    features = [
        column
        for column in data.select_dtypes(include="number").columns
        if column != "is_true_anomaly"
    ]

    labels = data["is_true_anomaly"].astype(bool)

    results = []

    for algorithm in ALGORITHMS:
        start = time.perf_counter()

        prediction = detect_anomalies(
            data,
            features=features,
            contamination=0.04,
            algorithm=algorithm,
        )

        elapsed = time.perf_counter() - start

        detected = prediction["is_anomaly"].astype(bool)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            detected,
            labels=[False, True],
        ).ravel()

        results.append(
            {
                "dataset": name,
                "algorithm": algorithm,
                "precision": float(precision_score(labels, detected, zero_division=0)),
                "recall": float(recall_score(labels, detected, zero_division=0)),
                "f1_score": float(f1_score(labels, detected, zero_division=0)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "detected_anomalies": int(detected.sum()),
                "runtime_ms": float(elapsed * 1000),
            }
        )

    return results


def plot_heatmap(results):
    table = results.pivot(
        index="dataset",
        columns="algorithm",
        values="f1_score",
    ).reindex(
        index=list(DATASETS),
        columns=ALGORITHMS,
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    image = ax.imshow(
        table.to_numpy(),
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )

    ax.set_xticks(range(len(table.columns)))
    ax.set_yticks(range(len(table.index)))

    ax.set_xticklabels(
        table.columns,
        rotation=15,
        ha="right",
    )
    ax.set_yticklabels(table.index)

    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.5 else "black",
            )

    fig.colorbar(image, ax=ax, label="F1-score")
    ax.set_title("AnomalyWatch — Algorithm Comparison")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "f1_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for name, filename in DATASETS.items():
        print(f"Evaluating {name}...")

        all_results.extend(evaluate_dataset(name, filename))

    results = pd.DataFrame(all_results)

    results.to_csv(
        OUTPUT_DIR / "benchmark_results.csv",
        index=False,
    )

    (OUTPUT_DIR / "benchmark_results.json").write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    plot_heatmap(results)

    print("\nAnomalyWatch Benchmark Suite")
    print("-" * 60)

    print(
        results[
            [
                "dataset",
                "algorithm",
                "precision",
                "recall",
                "f1_score",
                "runtime_ms",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
