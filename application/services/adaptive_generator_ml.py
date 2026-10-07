import json
import random
from pathlib import Path

import joblib
import pandas as pd

from application.generators.cellular_automata import generate_map as generate_cellular
from application.generators.perlin_noise import generate_perlin_map
from application.services.evaluation_metrics import evaluar_mapa
from application.services.feature_extractor import extract_ml_feature_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "experiment" / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"


def _read_json(name):
    with (ARTIFACT_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def _load_model(generator):
    path = MODEL_DIR / f"best_{generator}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path.name}; genere los artefactos desde el notebook final")
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


def generate_adaptive_map_ml(generator, config, attempts=10, requested_complexity="medio"):
    model = _load_model(generator)
    feature_config = _read_json("feature_config.json")
    manifest = _read_json("experiment_manifest.json")
    features = feature_config["features"]
    encoding = feature_config["target_encoding"]
    target = requested_complexity.lower()
    if target not in encoding:
        raise ValueError("Complejidad ML debe ser bajo, medio o alto")

    rng = random.Random(config.get("seed"))
    candidates = []
    for _ in range(attempts):
        seed = rng.randint(1, 2_147_483_647)
        grid, generated_parameters = _candidate(generator, config, seed, target)
        feature_dict = extract_ml_feature_dict(grid)
        sample = pd.DataFrame([{name: feature_dict[name] for name in features}], columns=features)
        predicted_id = int(model.predict(sample)[0])
        inverse = {int(v): k for k, v in encoding.items()}
        predicted = inverse[predicted_id]
        confidence = None
        target_probability = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            class_to_pos = {int(cls): pos for pos, cls in enumerate(model.classes_)}
            target_probability = float(probabilities[class_to_pos[encoding[target]]])
            confidence = float(max(probabilities))
        candidates.append({
            "map": grid, "seed": seed, "predicted_complexity": predicted,
            "matches_request": predicted == target,
            "confidence": None if confidence is None else round(confidence, 4),
            "target_probability": round(target_probability, 4),
            "metrics": evaluar_mapa(grid), "ml_features": feature_dict,
            "generated_parameters": generated_parameters,
        })

    # Primero coincidencia con la complejidad solicitada; después probabilidad/confianza.
    best = max(candidates, key=lambda x: (x["matches_request"], x["target_probability"], x["confidence"] or 0.0))
    winner = manifest["winners"][generator]
    return {
        **best,
        "algorithm": generator,
        "requested_complexity": target,
        "selected_model": winner["model"],
        "model_metrics": winner["metrics"],
        "model_artifact": winner["artifact"],
        "attempts": attempts,
        "parameters": config,
    }
