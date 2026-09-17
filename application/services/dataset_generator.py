import csv
import json
import random
from pathlib import Path

from application.generators.cellular_automata import generate_map as generate_cellular
from application.generators.perlin_noise import generate_perlin_map
from application.services.evaluator import evaluate
from application.services.feature_extractor import FEATURE_NAMES, extract_feature_dict


def _configuration(generator, rng):
    width, height = rng.randint(30, 50), rng.randint(30, 50)
    seed = rng.randint(1, 2_147_483_647)
    if generator == "cellular":
        parameters = {"probability": round(rng.uniform(0.38, 0.58), 4),
                      "iterations": rng.randint(2, 7)}
    else:
        parameters = {"scale": round(rng.uniform(0.035, 0.12), 4),
                      "octaves": rng.randint(2, 6),
                      "persistence": round(rng.uniform(0.35, 0.7), 4),
                      "threshold": round(rng.uniform(0.42, 0.58), 4)}
    return width, height, seed, parameters


def _generate(generator, width, height, seed, parameters):
    if generator == "cellular":
        return generate_cellular(width, height, parameters["probability"],
                                 parameters["iterations"], seed)
    return generate_perlin_map(width, height, parameters["scale"],
                               parameters["octaves"], parameters["persistence"],
                               seed, parameters["threshold"])


def generate_dataset(output_dir, samples_per_generator=500, experiment_seed=20260909):
    output_dir = Path(output_dir)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(experiment_seed)
    rows = []

    for generator in ("cellular", "perlin"):
        generator_dir = maps_dir / generator
        generator_dir.mkdir(parents=True, exist_ok=True)
        for number in range(1, samples_per_generator + 1):
            width, height, seed, parameters = _configuration(generator, rng)
            grid = _generate(generator, width, height, seed, parameters)
            features = extract_feature_dict(grid)
            score = evaluate(features)
            map_id = f"{generator}-{number:04d}"
            relative_path = f"maps/{generator}/{map_id}.json"
            record = {
                "id": map_id, "generator": generator, "seed": seed,
                "dimensions": {"width": width, "height": height},
                "parameters": parameters,
                "features": {name: round(features[name], 6) for name in FEATURE_NAMES},
                "target_score": score, "map": grid,
            }
            with (output_dir / relative_path).open("w", encoding="utf-8") as file:
                json.dump(record, file, ensure_ascii=False, separators=(",", ":"))
            rows.append({
                "id": map_id, "generator": generator, "json_path": relative_path,
                "seed": seed, "width": width, "height": height,
                **{name: round(features[name], 6) for name in FEATURE_NAMES},
                "target_score": score,
            })

    columns = ["id", "generator", "json_path", "seed", "width", "height",
               *FEATURE_NAMES, "target_score"]
    with (output_dir / "dataset_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "experiment_seed": experiment_seed,
        "samples_per_generator": samples_per_generator,
        "total_maps": len(rows),
        "generators": ["cellular", "perlin"],
        "feature_names": FEATURE_NAMES,
        "map_encoding": {"0": "transitable", "1": "obstacle"},
    }
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return rows
