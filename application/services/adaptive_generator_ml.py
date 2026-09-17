import random
from pathlib import Path

import joblib
import pandas as pd

from application.generators.cellular_automata import generate_map as generate_cellular
from application.generators.perlin_noise import generate_perlin_map
from application.services.evaluation_metrics import evaluar_mapa
from application.services.feature_extractor import extract_feature_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "experiment" / "artifacts" / "models"


def _load_bundle(generator):
    path = MODEL_DIR / f"best_{generator}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Ejecuta el experimento para crear {path.name}")
    return joblib.load(path)


def _candidate(generator, config, seed):
    if generator == "cellular":
        return generate_cellular(config["width"], config["height"],
                                 config["probability"], config["iterations"], seed)
    return generate_perlin_map(config["width"], config["height"], config["scale"],
                               config["octaves"], config["persistence"], seed,
                               config["threshold"])


def generate_adaptive_map_ml(generator, config, attempts=10):
    bundle = _load_bundle(generator)
    rng = random.Random(config.get("seed"))
    best = None

    for _ in range(attempts):
        seed = rng.randint(1, 2_147_483_647)
        grid = _candidate(generator, config, seed)
        feature_dict = extract_feature_dict(grid)
        sample = pd.DataFrame([feature_dict], columns=bundle["feature_names"])
        predicted_score = float(bundle["model"].predict(sample)[0])
        result = {"map": grid, "seed": seed, "predicted_score": round(predicted_score, 4),
                  "metrics": evaluar_mapa(grid)}
        if best is None or result["predicted_score"] > best["predicted_score"]:
            best = result

    return {
        **best,
        "algorithm": generator,
        "selected_model": bundle["model_name"],
        "model_metrics": bundle["metrics"],
        "attempts": attempts,
        "parameters": config,
    }
