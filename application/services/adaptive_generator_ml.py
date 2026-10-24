import json
import random
from pathlib import Path

import joblib
import pandas as pd

from application.generators.cellular_automata import generate_map as generate_cellular
from application.generators.perlin_noise import generate_perlin_map
from application.services.evaluation_metrics import evaluar_mapa
from application.services.feature_extractor import extract_ml_feature_dict

import logging
from time import perf_counter

from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "experiment" / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"


def _read_json(name):
    with (ARTIFACT_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def _load_model(generator):
    logging.warning(
        "[HU-78] Estado cache ML: %s",
        _load_model.cache_info()
    )
    path = MODEL_DIR / f"best_{generator}.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path.name}; "
            "genere los artefactos desde el notebook final"
        )

    return joblib.load(path)


def _candidate(generator, config, seed, target):
    rng = random.Random(seed)
    if generator == "cellular":
        profiles = {
            "bajo": ((0.40, 0.45), (3, 4), [4, 5], [3, 4]),
            "medio": ((0.44, 0.50), (4, 6), [4], [3]),
            "alto": ((0.49, 0.56), (5, 8), [3, 4], [2, 3]),
        }
        fill, steps, births, deaths = profiles[target]
        probability = rng.uniform(*fill)
        iterations = rng.randint(steps[0], steps[1])
        birth_limit, death_limit = rng.choice(births), rng.choice(deaths)
        grid = generate_cellular(config["width"], config["height"], probability, iterations, seed, birth_limit, death_limit)
        return grid, {"probability": probability, "iterations": iterations, "birth_limit": birth_limit, "death_limit": death_limit}

    profiles = {
        "bajo": ((2, 3), (0.35, 0.48), (-0.025, 0.005)),
        "medio": ((3, 5), (0.45, 0.60), (-0.005, 0.025)),
        "alto": ((5, 7), (0.55, 0.72), (0.015, 0.055)),
    }
    base_threshold = {"bosque": 0.48, "desierto": 0.44, "ciudad": 0.53}.get(config.get("terrain_type"), 0.48)
    octaves_range, persistence_range, delta_range = profiles[target]
    octaves = rng.randint(octaves_range[0], octaves_range[1])
    persistence = rng.uniform(*persistence_range)
    threshold = min(0.70, max(0.30, base_threshold + rng.uniform(*delta_range)))
    grid = generate_perlin_map(config["width"], config["height"], 0.08, octaves, persistence, seed, threshold)
    return grid, {"octaves": octaves, "persistence": persistence, "threshold": threshold}


def generate_adaptive_map_ml(
    generator,
    config,
    attempts=10,
    requested_complexity="medio"
):
    # ==========================================
    # HU-78: Diagnostico de rendimiento
    # ==========================================
    start_total = perf_counter()

    if not isinstance(attempts, int) or not 1 <= attempts <= 50:
        raise ValueError("'attempts' debe estar entre 1 y 50")

    # ==========================================
    # 1. Carga del modelo Machine Learning
    # ==========================================
    start_model = perf_counter()

    model = _load_model(generator)

    elapsed_model = (perf_counter() - start_model) * 1000

    logging.warning("[HU-78] Carga del modelo ML: %.2f ms", elapsed_model)
    logging.warning("[HU-78] Estado cache ML: %s", _load_model.cache_info())

    # ==========================================
    # 2. Carga de archivos de configuracion
    # ==========================================
    start_config = perf_counter()

    feature_config = _read_json("feature_config.json")
    manifest = _read_json("experiment_manifest.json")

    features = feature_config["features"]
    encoding = feature_config["target_encoding"]

    target = requested_complexity.lower()

    if target not in encoding:
        raise ValueError("Complejidad ML debe ser bajo, medio o alto")

    elapsed_config = (perf_counter() - start_config) * 1000

    logging.warning(
        "[HU-78] Carga de configuracion: %.2f ms",
        elapsed_config
    )

    # ==========================================
    # 3. Inicializacion de candidatos
    # ==========================================
    rng = random.Random(config.get("seed"))
    candidates = []

    # Acumuladores de tiempos
    time_generation = 0.0
    time_features = 0.0
    time_prediction = 0.0
    time_evaluation = 0.0

    inverse = {int(v): k for k, v in encoding.items()}

    # ==========================================
    # 4. Generacion y evaluacion de candidatos
    # ==========================================
    start_candidates = perf_counter()

    for _ in range(attempts):

        seed = rng.randint(1, 2_147_483_647)

        # --------------------------------------
        # 4.1 Generacion procedural
        # --------------------------------------
        start = perf_counter()

        grid, generated_parameters = _candidate(
            generator,
            config,
            seed,
            target
        )

        time_generation += (perf_counter() - start) * 1000

        # --------------------------------------
        # 4.2 Extraccion de caracteristicas
        # --------------------------------------
        start = perf_counter()

        feature_dict = extract_ml_feature_dict(grid)

        sample = pd.DataFrame(
            [{name: feature_dict[name] for name in features}],
            columns=features
        )

        time_features += (perf_counter() - start) * 1000

        # --------------------------------------
        # 4.3 Prediccion Machine Learning
        # --------------------------------------
        start = perf_counter()

        predicted_id = int(model.predict(sample)[0])

        predicted = inverse[predicted_id]

        confidence = None
        target_probability = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(sample)[0]

            class_to_pos = {
                int(cls): pos
                for pos, cls in enumerate(model.classes_)
            }

            target_probability = float(
                probabilities[
                    class_to_pos[encoding[target]]
                ]
            )

            confidence = float(max(probabilities))

        time_prediction += (perf_counter() - start) * 1000

        # --------------------------------------
        # 4.4 Evaluacion del mapa
        # --------------------------------------
        start = perf_counter()

        metrics = evaluar_mapa(
            grid,
            ml_features=feature_dict
        )

        time_evaluation += (perf_counter() - start) * 1000

        # --------------------------------------
        # 4.5 Almacenamiento del candidato
        # --------------------------------------
        candidates.append({
            "map": grid,
            "seed": seed,
            "predicted_complexity": predicted,
            "matches_request": predicted == target,
            "confidence": (
                None
                if confidence is None
                else round(confidence, 4)
            ),
            "target_probability": round(
                target_probability, 4
            ),
            "metrics": metrics,
            "ml_features": feature_dict,
            "generated_parameters": generated_parameters,
        })

    elapsed_candidates = (
        perf_counter() - start_candidates
    ) * 1000

    # ==========================================
    # 5. Seleccion del mejor mapa
    # ==========================================
    start_selection = perf_counter()

    best = max(
        candidates,
        key=lambda x: (
            x["matches_request"],
            x["target_probability"],
            x["confidence"] or 0.0
        )
    )

    winner = manifest["winners"][generator]

    elapsed_selection = (
        perf_counter() - start_selection
    ) * 1000

    # ==========================================
    # 6. Resultado final
    # ==========================================
    result = {
        **best,
        "algorithm": generator,
        "requested_complexity": target,
        "selected_model": winner["model"],
        "model_metrics": winner["metrics"],
        "model_artifact": winner["artifact"],
        "attempts": attempts,
        "parameters": config,
    }

    elapsed_total = (
        perf_counter() - start_total
    ) * 1000

    # ==========================================
    # 7. Reporte de diagnostico HU-78
    # ==========================================
    logging.warning("[HU-78] ===================================")
    logging.warning("[HU-78] Algoritmo: %s | Intentos: %s", generator, attempts)
    logging.warning("[HU-78] Carga modelo ML: %.2f ms", elapsed_model)
    logging.warning("[HU-78] Carga configuracion: %.2f ms", elapsed_config)
    logging.warning("[HU-78] Generacion procedural: %.2f ms",time_generation)
    logging.warning("[HU-78] Extraccion caracteristicas: %.2f ms", time_features)
    logging.warning("[HU-78] Prediccion ML: %.2f ms", time_prediction)
    logging.warning("[HU-78] Evaluacion mapas: %.2f ms", time_evaluation)
    logging.warning("[HU-78] Total ciclo candidatos: %.2f ms", elapsed_candidates)
    logging.warning("[HU-78] Seleccion mejor mapa: %.2f ms", elapsed_selection)
    logging.warning("[HU-78] TOTAL SERVICIO: %.2f ms", elapsed_total)
    logging.warning("[HU-78] ===================================")

    return result
