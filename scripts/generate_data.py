from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_data(n=500, seed=42):
    rng = np.random.default_rng(seed)

    temperature = rng.normal(45, 2, n)
    vibration = rng.normal(0.8, 0.12, n)
    pressure = rng.normal(100, 3, n)

    # Inject abnormal operating conditions.
    anomaly_indices = rng.choice(
        n, size=20, replace=False
    )

    temperature[anomaly_indices] += rng.uniform(
        15, 30, len(anomaly_indices)
    )
    vibration[anomaly_indices] += rng.uniform(
        0.8, 1.5, len(anomaly_indices)
    )
    pressure[anomaly_indices] -= rng.uniform(
        15, 25, len(anomaly_indices)
    )

    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01", periods=n, freq="min"
        ),
        "temperature_C": temperature,
        "vibration": vibration,
        "pressure_kPa": pressure,
        "injected_anomaly": [
            i in set(anomaly_indices)
            for i in range(n)
        ],
    })


if __name__ == "__main__":
    output = Path("data")
    output.mkdir(exist_ok=True)

    df = generate_sensor_data()
    df.to_csv(output / "sample_sensors.csv", index=False)

    print(f"Generated {len(df)} observations")
    print("Saved: data/sample_sensors.csv")
