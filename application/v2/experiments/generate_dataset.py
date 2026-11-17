import argparse
import csv
import math
from pathlib import Path

from application.v2.services.map_service import generate
from application.v2.services.ml_adapter import collision_to_ml_grid
from application.services.feature_extractor import extract_ml_feature_dict

BIOMES = ("bosque", "desierto", "ciudad", "mazmorra")
COMPLEXITIES = ("bajo", "medio", "alto")

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

FIELDS = (
    "biome",
    "complexity_requested",
    "seed",
    "variant",
    *FEATURES,
)


def build_distribution(total):
    """Distribuye el total entre las 12 combinaciones."""
    combinations = [
        (biome, complexity)
        for biome in BIOMES
        for complexity in COMPLEXITIES
    ]

    base, remainder = divmod(total, len(combinations))

    return [
        (biome, complexity, base + (index < remainder))
        for index, (biome, complexity) in enumerate(combinations)
    ]



def extract_features(result):
    """
    Extrae las características ML a partir de
    la capa collision del mapa V2.
    """

    layers = result.get("layers")

    if not isinstance(layers, dict):
        raise ValueError("La respuesta no contiene layers")

    collision = layers.get("collision")

    if collision is None:
        raise ValueError("La respuesta no contiene collision")

    # Convierte la matriz V2 al formato utilizado por ML.
    grid_ml = collision_to_ml_grid(collision)

    # Utiliza el extractor original de características.
    feature_dict = extract_ml_feature_dict(grid_ml)

    missing = [
        name for name in FEATURES
        if name not in feature_dict
    ]

    if missing:
        raise ValueError(
            f"Características faltantes: {missing}"
        )

    values = {}

    for name in FEATURES:
        value = float(feature_dict[name])

        if not math.isfinite(value):
            raise ValueError(
                f"Característica no finita: {name}={value}"
            )

        values[name] = value

    return values



def generate_dataset(
        total=120,
        output="application/v2/experiments/results/dataset_v2.csv",
        base_seed=12345,
):
    if total < 12:
        raise ValueError("El total debe ser al menos 12.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distribution = build_distribution(total)
    generated = 0

    print(f"Iniciando generación de {total} mapas V2...")
    print(f"Destino: {output_path.resolve()}")

    with output_path.open(
            "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for biome, complexity, count in distribution:
            print(f"Generando {biome} / {complexity}: {count}")

            for index in range(count):
                # Se reutiliza la misma secuencia de semillas
                # entre perfiles para permitir comparaciones.
                seed = base_seed + index

                payload = {
                    "biome": biome,
                    "complexity": complexity,
                    "seed": seed,
                    "width": 50,
                    "height": 35,
                    "variant": "B",
                    "include_ml": False,
                }

                result = generate(payload)
                features = extract_features(result)

                features = extract_features(result)

                writer.writerow({
                    "biome": biome,
                    "complexity_requested": complexity,
                    "seed": seed,
                    "variant": "B",
                    **features,
                })

                generated += 1

                if generated % 100 == 0:
                    print(f"Progreso: {generated}/{total}")

    print(f"Dataset generado: {generated} registros")
    print(f"Archivo: {output_path.resolve()}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generación del dataset experimental Rhyzome V2"
    )

    parser.add_argument("--total", type=int, default=120)
    parser.add_argument(
        "--output",
        default="application/v2/experiments/results/dataset_v2.csv",
    )
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    generate_dataset(
        total=args.total,
        output=args.output,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
