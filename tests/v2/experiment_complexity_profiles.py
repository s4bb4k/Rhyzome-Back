"""Evaluación experimental de perfiles de Rhyzome V2."""

import csv
from pathlib import Path

import numpy as np

from application.v2.services.map_service import generate

BIOMES = ("bosque", "desierto", "ciudad", "mazmorra")
COMPLEXITIES = ("bajo", "medio", "alto")
SEEDS = range(12345, 12375)

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


def run_experiment():
    records = []

    for biome in BIOMES:
        for complexity in COMPLEXITIES:
            for seed in SEEDS:

                result = generate({
                    "terrain_type": biome,
                    "width": 50,
                    "height": 35,
                    "seed": seed,
                    "variant": "B",
                    "complexity": complexity,
                    "include_ml": True,
                })

                ml = result["ml"]
                features = ml["ml_features"]

                record = {
                    "biome": biome,
                    "complexity_requested": complexity,
                    "complexity_predicted": ml[
                        "predicted_complexity"
                    ],
                    "seed": seed,
                    "match": int(
                        ml["predicted_complexity"]
                        == complexity
                    ),
                }

                for feature in FEATURES:
                    record[feature] = features[feature]

                records.append(record)

    return records


def print_summary(records):
    print("\nRESULTADOS EXPERIMENTALES RHYZOME V2")
    print("=" * 70)

    for biome in BIOMES:
        print(f"\nBIOMA: {biome.upper()}")

        for complexity in COMPLEXITIES:
            subset = [
                r for r in records
                if r["biome"] == biome
                   and r["complexity_requested"] == complexity
            ]

            accuracy = np.mean([
                r["match"] for r in subset
            ])

            floor_ratio = np.mean([
                r["floor_ratio"] for r in subset
            ])

            edge_density = np.mean([
                r["edge_density"] for r in subset
            ])

            print(
                f"{complexity:6} | "
                f"Coincidencias: {accuracy:.2%} | "
                f"Floor ratio: {floor_ratio:.4f} | "
                f"Edge density: {edge_density:.4f}"
            )


def save_csv(records):
    output_dir = Path("reports/v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
            output_dir / "complexity_profiles_experiment.csv"
    )

    with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nReporte generado: {output_file}")


if __name__ == "__main__":
    records = run_experiment()

    print_summary(records)
    save_csv(records)
