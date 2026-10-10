from pathlib import Path

import numpy as np
import pandas as pd


def generate_benchmarks(n=500, seed=42):
    rng = np.random.default_rng(seed)

    output = Path("data/benchmarks")
    output.mkdir(parents=True, exist_ok=True)

    # 1. Point anomalies
    normal = rng.normal(size=(n, 3))
    labels = np.zeros(n, dtype=bool)

    indices = rng.choice(n, 20, replace=False)
    labels[indices] = True

    normal[indices] += 7.0

    pd.DataFrame({
        "sensor_1": normal[:, 0],
        "sensor_2": normal[:, 1],
        "sensor_3": normal[:, 2],
        "is_true_anomaly": labels,
    }).to_csv(output / "point_anomalies.csv", index=False)

    # 2. Gradual drift
    data = rng.normal(0, 0.3, (n, 3))
    labels = np.zeros(n, dtype=bool)

    drift_start = 350
    drift = np.linspace(0, 3.0, n - drift_start)

    data[drift_start:, 0] += drift

    # Label observations after meaningful drift.
    labels[drift_start + 50:] = True

    pd.DataFrame({
        "sensor_1": data[:, 0],
        "sensor_2": data[:, 1],
        "sensor_3": data[:, 2],
        "is_true_anomaly": labels,
    }).to_csv(output / "gradual_drift.csv", index=False)

    # 3. Contextual anomalies
    load = rng.uniform(0, 1, n)
    temperature = 30 + 40 * load + rng.normal(0, 2, n)
    vibration = 0.3 + 0.5 * load + rng.normal(0, 0.05, n)

    labels = np.zeros(n, dtype=bool)
    indices = rng.choice(n, 20, replace=False)
    labels[indices] = True

    # Break the temperature-load relationship.
    temperature[indices] = (
        70 - 40 * load[indices]
        + rng.normal(0, 2, len(indices))
    )

    pd.DataFrame({
        "load": load,
        "temperature": temperature,
        "vibration": vibration,
        "is_true_anomaly": labels,
    }).to_csv(output / "contextual_anomalies.csv", index=False)

    print(f"Benchmark datasets saved to {output}")


if __name__ == "__main__":
    generate_benchmarks()
