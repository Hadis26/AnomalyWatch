import pandas as pd
import pytest

from scripts.generate_benchmarks import generate_benchmarks
from scripts.benchmark_suite import (
    DATASETS,
    ALGORITHMS,
    evaluate_dataset,
)


def test_benchmark_suite(tmp_path, monkeypatch):
    # Generate data in an isolated temporary directory.
    monkeypatch.chdir(tmp_path)

    generate_benchmarks(n=500, seed=42)

    # The benchmark module resolves its data directory
    # relative to the repository, so redirect it here.
    import scripts.benchmark_suite as suite

    monkeypatch.setattr(
        suite,
        "DATA_DIR",
        tmp_path / "data" / "benchmarks",
    )

    for dataset_name, filename in DATASETS.items():
        results = evaluate_dataset(
            dataset_name,
            filename,
        )

        assert len(results) == len(ALGORITHMS)

        for result in results:
            assert 0 <= result["precision"] <= 1
            assert 0 <= result["recall"] <= 1
            assert 0 <= result["f1_score"] <= 1
            assert result["runtime_ms"] >= 0

            total = (
                result["true_positives"]
                + result["false_positives"]
                + result["false_negatives"]
                + result["true_negatives"]
            )

            assert total == 500