"""Adaptador ML experimental para Rhyzome V2."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from application.services.feature_extractor import (
    extract_ml_feature_dict,
)

MODEL_PATH = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "ml_v2"
        / "best_model_v2.joblib"
)

FEATURES = (
    "floor_ratio",
    "n_regions",
    "largest_region_ratio",
    "edge_density",
    "row_entropy",
    "col_entropy",
    "local_variance",
    "perimeter_area_ratio",
)

CLASS_ENCODING = {
    "bajo": 0,
    "medio": 1,
    "alto": 2,
}

CLASS_DECODING = {
    value: key
    for key, value in CLASS_ENCODING.items()
}


def collision_to_ml_grid(collision):
    """Convierte collision V2 al formato ML."""

    collision = np.asarray(collision)

    if collision.ndim != 2 or collision.size == 0:
        raise ValueError(
            "collision debe ser una matriz 2D no vacía"
        )

    if not np.isin(collision, [0, 1]).all():
        raise ValueError(
            "collision solo puede contener 0 y 1"
        )

    # V2: 0 transitable, 1 obstáculo.
    # ML: 1 transitable, 0 obstáculo.
    return (collision == 0).astype(np.uint8)


@lru_cache(maxsize=1)
def load_v2_model():
    """Carga una sola vez el modelo experimental V2."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo V2: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise ValueError(
            "El archivo no contiene un clasificador válido"
        )

    return model


def predict_v2_complexity(map_data, generator):
    """
    Clasifica el perfil de generación de un mapa V2.

    Conserva el contrato de respuesta del adaptador
    utilizado por map_service y adaptive_service.
    """

    if generator not in ("perlin", "cellular"):
        raise ValueError("Generador no válido")

    if not isinstance(map_data, dict):
        raise ValueError("map_data debe ser un diccionario")

    if "collision" not in map_data:
        raise ValueError("El mapa no contiene collision")

    grid_ml = collision_to_ml_grid(map_data["collision"])

    feature_dict = extract_ml_feature_dict(grid_ml)

    missing = [
        name for name in FEATURES
        if name not in feature_dict
    ]

    if missing:
        raise ValueError(
            f"Características faltantes: {missing}"
        )

    sample = pd.DataFrame(
        [{
            name: float(feature_dict[name])
            for name in FEATURES
        }],
        columns=list(FEATURES),
    )

    if not np.isfinite(
            sample.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Las características contienen valores no finitos"
        )

    model = load_v2_model()

    predicted_id = int(model.predict(sample)[0])

    if predicted_id not in CLASS_DECODING:
        raise ValueError(
            f"Clase predicha desconocida: {predicted_id}"
        )

    predicted = CLASS_DECODING[predicted_id]

    class_probabilities = {
        label: None
        for label in CLASS_ENCODING
    }

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]

        if not hasattr(model, "classes_"):
            raise ValueError(
                "El modelo no expone classes_"
            )

        classes = [
            int(value)
            for value in model.classes_
        ]

        if len(classes) != len(probabilities):
            raise ValueError(
                "Inconsistencia entre clases y probabilidades"
            )

        for class_id, probability in zip(
                classes, probabilities
        ):
            if class_id in CLASS_DECODING:
                label = CLASS_DECODING[class_id]
                class_probabilities[label] = float(
                    probability
                )

        confidence = class_probabilities[predicted]

    return {
        "predicted_complexity": predicted,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "ml_features": {
            name: float(sample.iloc[0][name])
            for name in FEATURES
        },
        "model_generator": generator,
    }
