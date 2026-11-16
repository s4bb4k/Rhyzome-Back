"""Entrenamiento experimental de clasificadores de perfiles para Rhyzome V2.

Ejecutar desde la raíz del backend:
    python -m application.v2.experiments.train_models

Las etiquetas representan perfiles de generación, no dificultad jugable validada.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = [
    "floor_ratio",
    "n_regions",
    "largest_region_ratio",
    "edge_density",
    "row_entropy",
    "col_entropy",
    "local_variance",
    "perimeter_area_ratio",
]

CLASSES = ["bajo", "medio", "alto"]
CLASS_ENCODING = {"bajo": 0, "medio": 1, "alto": 2}
CLASS_DECODING = {value: key for key, value in CLASS_ENCODING.items()}
CLASS_IDS = [CLASS_ENCODING[label] for label in CLASSES]


def split_dataset(df):
    """Partición reproducible por semilla: ~70 % / 15 % / 15 %."""
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(first.split(df, groups=df["seed"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, test_idx = next(second.split(temp_df, groups=temp_df["seed"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    seed_sets = [set(part["seed"]) for part in (train_df, val_df, test_df)]
    if (not seed_sets[0].isdisjoint(seed_sets[1])
            or not seed_sets[0].isdisjoint(seed_sets[2])
            or not seed_sets[1].isdisjoint(seed_sets[2])):
        raise RuntimeError("Fuga de información: semillas compartidas entre particiones")

    return train_df, val_df, test_df


def get_models():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),

        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }


def evaluate(model, x, y):
    predictions = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(
            y, predictions, labels=CLASS_IDS,
            average="macro", zero_division=0,
        )),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar y evaluar clasificadores de perfiles Rhyzome V2"
    )
    parser.add_argument(
        "--dataset",
        default="application/v2/experiments/results/dataset_v2_10000.csv",
    )
    parser.add_argument(
        "--output",
        default="application/v2/experiments/results/ml_v2",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    required = FEATURES + ["seed", "biome", "complexity_requested"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")
    if df[required].isnull().any().any():
        raise ValueError("El dataset contiene valores nulos")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("El dataset contiene características no finitas")
    if not df["complexity_requested"].isin(CLASSES).all():
        raise ValueError("El dataset contiene etiquetas desconocidas")
    if df["seed"].nunique() < 3:
        raise ValueError("Se necesitan al menos tres semillas distintas")

    train_df, val_df, test_df = split_dataset(df)
    partitions = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }
    print("=== PARTICIÓN DEL DATASET ===")
    partition_summary = {}
    for name, part in partitions.items():
        counts = part["complexity_requested"].value_counts().to_dict()
        if set(counts) != set(CLASSES):
            raise ValueError(f"Faltan clases en la partición {name}: {counts}")
        partition_summary[name] = {
            "records": int(len(part)),
            "unique_seeds": int(part["seed"].nunique()),
            "class_counts": {label: int(counts[label]) for label in CLASSES},
        }
        print(f"{name}: {len(part)} registros, {part['seed'].nunique()} semillas; clases={counts}")

    x_train = train_df[FEATURES]
    y_train = train_df["complexity_requested"].map(CLASS_ENCODING).astype(int)
    x_val = val_df[FEATURES]
    y_val = val_df["complexity_requested"].map(CLASS_ENCODING).astype(int)
    x_test = test_df[FEATURES]
    y_test = test_df["complexity_requested"].map(CLASS_ENCODING).astype(int)

    results = {}
    trained = {}
    print("\n=== ENTRENAMIENTO ===")
    for name, model in get_models().items():
        print(f"\nEntrenando {name}...", flush=True)
        model.fit(x_train, y_train)
        train_metrics = evaluate(model, x_train, y_train)
        val_metrics = evaluate(model, x_val, y_val)
        results[name] = {"train": train_metrics, "validation": val_metrics}
        trained[name] = model
        print("Entrenamiento:", train_metrics)
        print("Validación:", val_metrics)

    best_name = max(
        trained,
        key=lambda name: results[name]["validation"]["macro_f1"],
    )
    best_model = trained[best_name]
    test_predictions = best_model.predict(x_test)
    test_metrics = evaluate(best_model, x_test, y_test)
    results[best_name]["test"] = test_metrics

    report = classification_report(
        y_test, test_predictions, labels=CLASS_IDS,
        target_names=CLASSES, zero_division=0, output_dict=True,
    )
    matrix = confusion_matrix(y_test, test_predictions, labels=CLASS_IDS)
    by_biome = {}
    for biome, indices in test_df.groupby("biome").groups.items():
        # Los índices de pandas pueden no ser consecutivos; usar .loc sobre las series.
        biome_y = y_test.loc[indices]
        biome_x = x_test.loc[indices]
        by_biome[str(biome)] = evaluate(best_model, biome_x, biome_y)
        by_biome[str(biome)]["records"] = int(len(indices))

    results["selected_model"] = best_name
    results["test_classification_report"] = report
    results["test_confusion_matrix"] = matrix.tolist()
    results["test_by_biome"] = by_biome
    results["class_order"] = CLASSES
    results["class_encoding"] = CLASS_ENCODING
    results["feature_order"] = FEATURES
    results["dataset_size"] = int(len(df))
    results["partitions"] = partition_summary
    results["label_semantics"] = "perfil_generativo_solicitado_no_dificultad_jugable_validada"

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "best_model_v2.joblib"
    results_path = output / "training_results.json"
    joblib.dump(best_model, model_path)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n=== RESULTADOS FINALES ===")
    print("Modelo seleccionado:", best_name)
    print("Accuracy prueba:", test_metrics["accuracy"])
    print("Macro F1 prueba:", test_metrics["macro_f1"])
    print("Matriz de confusión (bajo, medio, alto):\n", matrix)
    print("Resultados por bioma:", by_biome)
    print("Modelo guardado:", model_path)
    print("Resultados guardados:", results_path)


if __name__ == "__main__":
    main()
