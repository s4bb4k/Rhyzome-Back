"""Servicio de generación de mapas Rhyzome V2.

Responsabilidades:
- Validar los parámetros de generación.
- Seleccionar el bioma y la variante A/B.
- Aplicar perfiles experimentales de complejidad.
- Generar las capas semánticas del mapa.
- Calcular métricas estructurales.
- Ejecutar predicción ML cuando se solicite.

IMPORTANTE:
- Los generadores V1 permanecen independientes.
- El ML V1 no se invoca automáticamente.
- Los perfiles V2 todavía requieren validación experimental.
"""

import secrets

from application.v2.generators.semantic import (
    generar_A,
    TERRAIN,
    OBJECT,
    BIOMAS,
)

from application.v2.generators.optimized import generar_B

from application.v2.services.metrics import metricas

from application.v2.services.ml_adapter import (
    predict_v2_complexity,
)

from application.v2.services.complexity_profiles import (
    get_complexity_profile,
)

ALIASES = {
    "forest": "bosque",
    "desert": "desierto",
    "urban": "ciudad",
    "urbano": "ciudad",
    "dungeon": "mazmorra",
}

COMPLEXITIES = ("bajo", "medio", "alto")


def _int(data, key, default, minimum, maximum):
    """Valida un parámetro entero dentro del rango permitido."""

    raw = data.get(key, default)

    if isinstance(raw, bool):
        raise ValueError(
            f"'{key}' debe ser entero"
        )

    try:
        value = int(raw)

        if str(raw).strip() != str(value):
            raise ValueError()

    except (TypeError, ValueError):
        raise ValueError(
            f"'{key}' debe ser entero"
        ) from None

    if not minimum <= value <= maximum:
        raise ValueError(
            f"'{key}' debe estar entre {minimum} y {maximum}"
        )

    return value


def _validate_complexity(data):
    """Valida la complejidad experimental solicitada.

    Returns:
        None cuando no se solicita complejidad.
        'bajo', 'medio' o 'alto' cuando es válida.
    """

    complexity = data.get("complexity")

    if complexity is None:
        return None

    if not isinstance(complexity, str):
        raise ValueError(
            "'complexity' debe ser bajo, medio o alto"
        )

    complexity = complexity.strip().lower()

    if complexity not in COMPLEXITIES:
        raise ValueError(
            "'complexity' debe ser bajo, medio o alto"
        )

    return complexity


def generate(data):
    """Genera un mapa semántico Rhyzome V2.

    Parámetros:
        terrain_type: bosque, desierto, ciudad o mazmorra.
        width: anchura del mapa, entre 10 y 200.
        height: altura del mapa, entre 10 y 200.
        seed: semilla opcional.
        variant: A o B, por defecto B.
        complexity: bajo, medio o alto (opcional).
        include_ml: habilita la predicción ML (opcional).

    Returns:
        Diccionario con el mapa, sus métricas,
        las capas semánticas y los resultados
        experimentales solicitados.
    """

    # ==================================================
    # 1. VALIDACIÓN DEL CUERPO JSON
    # ==================================================

    if not isinstance(data, dict):
        raise ValueError(
            "El cuerpo debe ser un objeto JSON"
        )

    # ==================================================
    # 2. VALIDACIÓN DE MACHINE LEARNING
    # ==================================================

    include_ml = data.get("include_ml", False)

    if not isinstance(include_ml, bool):
        raise ValueError(
            "'include_ml' debe ser booleano"
        )

    # ==================================================
    # 3. VALIDACIÓN DEL BIOMA
    # ==================================================

    biome = str(
        data.get(
            "terrain_type",
            data.get("biome", "bosque"),
        )
    ).lower().strip()

    biome = ALIASES.get(biome, biome)

    if biome not in BIOMAS:
        raise ValueError(
            "'terrain_type' debe ser bosque, "
            "desierto, ciudad o mazmorra"
        )

    # ==================================================
    # 4. VALIDACIÓN DE DIMENSIONES Y SEMILLA
    # ==================================================

    width = _int(
        data,
        "width",
        50,
        10,
        200,
    )

    height = _int(
        data,
        "height",
        35,
        10,
        200,
    )

    seed = _int(
        data,
        "seed",
        secrets.randbelow(2147483648),
        0,
        2147483647,
    )

    # ==================================================
    # 5. VALIDACIÓN DE VARIANTE
    # ==================================================

    version = str(
        data.get("variant", "B")
    ).upper()

    if version not in ("A", "B"):
        raise ValueError(
            "'variant' debe ser A o B"
        )

    # ==================================================
    # 6. PERFIL EXPERIMENTAL DE COMPLEJIDAD
    # ==================================================

    complexity = _validate_complexity(data)

    profile = (
        get_complexity_profile(biome, complexity)
        if complexity is not None
        else None
    )

    # ==================================================
    # 7. GENERACIÓN DEL MAPA
    # ==================================================

    if version == "A":
        m = generar_A(
            biome,
            seed,
            height,
            width,
            profile=profile,
        )

        cleared = 0

    else:
        m, cleared = generar_B(
            biome,
            seed,
            height,
            width,
            profile=profile,
        )

    # ==================================================
    # 8. CONSTRUCCIÓN DE RESPUESTA
    # ==================================================

    payload = {
        "status": "success",
        "version": "v2",
        "schema_version": "2.1-experimental",
        "variant": version,
        "terrain_type": biome,
        "biome": biome,
        "seed": seed,
        "width": width,
        "height": height,
        "dimensions": {
            "width": width,
            "height": height,
        },
        "terrain_catalog": TERRAIN,
        "object_catalog": OBJECT,
        "layers": {
            key: m[key].astype(int).tolist()
            for key in (
                "terrain",
                "objects",
                "collision",
            )
        },
        "metrics": metricas(m),
        "repaired_cells": cleared,
        "generator_note": (
            "ruido_suave experimental; "
            "no equivale al Perlin de V1"
        ),
    }

    # ==================================================
    # 9. TRAZABILIDAD DEL PERFIL UTILIZADO
    # ==================================================

    if complexity is not None:
        payload["generation_profile"] = {
            "complexity_requested": complexity,
            "parameters": profile,
            "mode": "experimental_v2",
            "validated": False,
        }

    # ==================================================
    # 10. PREDICCIÓN EXPERIMENTAL ML
    # ==================================================

    if include_ml:
        model_generator = (
            "cellular"
            if biome == "mazmorra"
            else "perlin"
        )

        prediction = predict_v2_complexity(
            payload["layers"],
            model_generator,
        )

        payload["ml"] = {
            **prediction,
            "evaluation_mode": "experimental_v2",
            "model_selection": "biome_proxy",
            "compatibility_validated": False,
        }

    return payload
