"""Perfiles experimentales de complejidad para Rhyzome V2.

Los valores son hipótesis iniciales de generación.
No representan complejidades validadas por Machine Learning.
"""

COMPLEXITY_PROFILES = {
    "bosque": {
        "bajo": {
            "water_threshold": 0.10,
            "tree_probability": 0.15,
            "rock_probability": 0.015,
        },
        "medio": {
            "water_threshold": 0.17,
            "tree_probability": 0.31,
            "rock_probability": 0.035,
        },
        "alto": {
            "water_threshold": 0.24,
            "tree_probability": 0.45,
            "rock_probability": 0.06,
        },
    },
    "desierto": {
        "bajo": {
            "water_threshold": 0.06,
            "cactus_probability": 0.04,
            "rock_probability": 0.02,
        },
        "medio": {
            "water_threshold": 0.12,
            "cactus_probability": 0.09,
            "rock_probability": 0.04,
        },
        "alto": {
            "water_threshold": 0.18,
            "cactus_probability": 0.16,
            "rock_probability": 0.07,
        },
    },
    "ciudad": {
        "bajo": {
            "building_probability": 0.45,
            "tree_probability": 0.01,
        },
        "medio": {
            "building_probability": 0.73,
            "tree_probability": 0.025,
        },
        "alto": {
            "building_probability": 0.90,
            "tree_probability": 0.05,
        },
    },
    "mazmorra": {
        "bajo": {
            "wall_probability": 0.38,
            "iterations": 3,
        },
        "medio": {
            "wall_probability": 0.43,
            "iterations": 4,
        },
        "alto": {
            "wall_probability": 0.49,
            "iterations": 6,
        },
    },
}


def get_complexity_profile(biome, complexity):
    """Devuelve una copia del perfil solicitado."""

    if biome not in COMPLEXITY_PROFILES:
        raise ValueError(f"Bioma no válido: {biome}")

    if complexity not in ("bajo", "medio", "alto"):
        raise ValueError(
            "La complejidad debe ser bajo, medio o alto"
        )

    return COMPLEXITY_PROFILES[biome][complexity].copy()
