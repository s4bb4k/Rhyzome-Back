"""Diagnóstico experimental de complejidad Rhyzome V2."""

from application.v2.services.map_service import generate

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


def main():
    for biome in BIOMES:
        print(f"\n{'=' * 75}")
        print(f"BIOMA: {biome.upper()}")
        print("=" * 75)

        for complexity in COMPLEXITIES:
            result = generate({
                "terrain_type": biome,
                "width": 50,
                "height": 35,
                "seed": 12345,
                "variant": "B",
                "complexity": complexity,
                "include_ml": True,
            })

            ml = result["ml"]
            features = ml["ml_features"]

            print(f"\nPerfil solicitado: {complexity}")
            print(
                "Predicción ML:",
                ml["predicted_complexity"],
            )
            print(
                "Confianza:",
                ml["confidence"],
            )

            for feature in FEATURES:
                print(
                    f"  {feature}: "
                    f"{features[feature]:.6f}"
                )


if __name__ == "__main__":
    main()
