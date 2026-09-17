import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from application.services.dataset_generator import generate_dataset
from application.services.feature_extractor import FEATURE_NAMES


def _models():
    return {
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100.0, epsilon=0.05, gamma="scale")),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                learning_rate_init=0.001, max_iter=2000, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=40, random_state=42,
            )),
        ]),
    }


def run_experiment(project_root, samples_per_generator=500):
    project_root = Path(project_root)
    dataset_dir = project_root / "experiment" / "dataset"
    artifacts_dir = project_root / "experiment" / "artifacts"
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    index_path = dataset_dir / "dataset_index.csv"
    if index_path.exists():
        dataframe = pd.read_csv(index_path)
    else:
        dataframe = pd.DataFrame(generate_dataset(
            dataset_dir, samples_per_generator=samples_per_generator
        ))
    for previous_model in models_dir.glob("*.joblib"):
        previous_model.unlink()
    results = []

    for generator in ("cellular", "perlin"):
        subset = dataframe[dataframe["generator"] == generator]
        train, test = train_test_split(subset, test_size=0.2, random_state=42)
        split = {"generator": generator, "train_ids": train["id"].tolist(),
                 "test_ids": test["id"].tolist()}
        with (artifacts_dir / f"split_{generator}.json").open("w", encoding="utf-8") as file:
            json.dump(split, file, ensure_ascii=False, indent=2)

        generator_results = []
        for name, estimator in _models().items():
            estimator.fit(train[FEATURE_NAMES], train["target_score"])
            predictions = estimator.predict(test[FEATURE_NAMES])
            metrics = {
                "generator": generator, "model": name,
                "train_samples": len(train), "test_samples": len(test),
                "r2": round(r2_score(test["target_score"], predictions), 6),
                "mae": round(mean_absolute_error(test["target_score"], predictions), 6),
                "rmse": round(math.sqrt(mean_squared_error(test["target_score"], predictions)), 6),
            }
            model_path = models_dir / f"{generator}_{name}.joblib"
            joblib.dump({"model": estimator, "feature_names": FEATURE_NAMES,
                         "generator": generator, "model_name": name,
                         "metrics": metrics}, model_path)
            metrics["artifact"] = f"models/{model_path.name}"
            generator_results.append(metrics)
            results.append(metrics)

        best = max(generator_results, key=lambda item: (item["r2"], -item["mae"]))
        shutil.copyfile(models_dir / f"{generator}_{best['model']}.joblib",
                        models_dir / f"best_{generator}.joblib")

    winners = {
        generator: max((item for item in results if item["generator"] == generator),
                       key=lambda item: (item["r2"], -item["mae"]))["model"]
        for generator in ("cellular", "perlin")
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_maps": len(dataframe),
        "samples_per_generator": int(len(dataframe) / 2),
        "selection_rule": "Mayor R²; menor MAE como desempate",
        "results": results, "winners": winners,
    }
    with (artifacts_dir / "experiment_results.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(artifacts_dir / "model_comparison.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento reproducible de Rhizome")
    parser.add_argument("--samples-per-generator", type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run_experiment(root, args.samples_per_generator), indent=2))
