
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from application.services import train_model as module
from application.services.feature_extractor import FEATURE_NAMES


# Clase simulada definida a nivel de módulo.
# Esto permite que joblib pueda serializarla.
class FakeEstimator:

    def fit(self, x, y):
        self.value = float(y.mean())
        return self

    def predict(self, x):
        return [getattr(self, "value", 0.0)] * len(x)


def create_fake_rows():
    rows = []

    for generator in ("cellular", "perlin"):
        for index in range(10):
            rows.append({
                "id": f"{generator}-{index}",
                "generator": generator,
                "target_score": float(index),
                **{
                    name: float(index)
                    for name in FEATURE_NAMES
                }
            })

    return rows


def test_models_have_expected_names():
    models = module._models()

    assert set(models) == {
        "random_forest",
        "svm",
        "mlp"
    }

    assert all(
        hasattr(estimator, "fit")
        and hasattr(estimator, "predict")
        for estimator in models.values()
    )


def test_run_experiment_with_fake_estimators(
    tmp_path,
    monkeypatch
):
    rows = create_fake_rows()

    def fake_dataset(output_dir, samples_per_generator):
        return rows

    monkeypatch.setattr(
        module,
        "generate_dataset",
        fake_dataset
    )

    monkeypatch.setattr(
        module,
        "_models",
        lambda: {
            "random_forest": FakeEstimator(),
            "svm": FakeEstimator(),
            "mlp": FakeEstimator()
        }
    )

    summary = module.run_experiment(
        tmp_path,
        samples_per_generator=10
    )

    assert summary["total_maps"] == 20
    assert len(summary["results"]) == 6

    assert set(summary["winners"]) == {
        "cellular",
        "perlin"
    }

    artifacts = (
        tmp_path
        / "experiment"
        / "artifacts"
    )

    assert (
        artifacts / "experiment_results.json"
    ).exists()

    assert (
        artifacts / "model_comparison.csv"
    ).exists()

    for generator in ("cellular", "perlin"):

        assert (
            artifacts / f"split_{generator}.json"
        ).exists()

        model_file = (
            artifacts
            / "models"
            / f"best_{generator}.joblib"
        )

        assert model_file.exists()

        split = json.loads(
            (
                artifacts / f"split_{generator}.json"
            ).read_text(encoding="utf-8")
        )

        assert len(split["train_ids"]) == 8
        assert len(split["test_ids"]) == 2


def test_run_experiment_uses_existing_csv(
    tmp_path,
    monkeypatch
):
    dataset_dir = (
        tmp_path
        / "experiment"
        / "dataset"
    )

    dataset_dir.mkdir(parents=True)

    rows = create_fake_rows()

    pd.DataFrame(rows).to_csv(
        dataset_dir / "dataset_index.csv",
        index=False
    )

    monkeypatch.setattr(
        module,
        "generate_dataset",
        Mock(
            side_effect=AssertionError(
                "No se debe regenerar el dataset"
            )
        )
    )

    monkeypatch.setattr(
        module,
        "_models",
        lambda: {
            "fake": FakeEstimator()
        }
    )

    result = module.run_experiment(tmp_path)

    assert result["total_maps"] == 20
    assert len(result["results"]) == 2
