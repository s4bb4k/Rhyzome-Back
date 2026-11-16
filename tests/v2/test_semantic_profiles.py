import numpy as np
import pytest

from application.v2.generators.semantic import generar_A
from application.v2.services.complexity_profiles import (
    get_complexity_profile,
)

BIOMES = ("bosque", "desierto", "ciudad", "mazmorra")
COMPLEXITIES = ("bajo", "medio", "alto")
LAYERS = ("terrain", "objects", "collision")


@pytest.mark.parametrize("biome", BIOMES)
def test_perfil_medio_conserva_resultado_original(biome):
    original = generar_A(biome, 12345, 35, 50)

    profile = get_complexity_profile(biome, "medio")
    experimental = generar_A(
        biome, 12345, 35, 50, profile=profile
    )

    for layer in LAYERS:
        np.testing.assert_array_equal(
            original[layer],
            experimental[layer],
        )


@pytest.mark.parametrize("biome", BIOMES)
@pytest.mark.parametrize("complexity", COMPLEXITIES)
def test_generacion_con_perfil_es_reproducible(
        biome, complexity
):
    profile = get_complexity_profile(biome, complexity)

    first = generar_A(
        biome, 12345, 35, 50, profile=profile
    )
    second = generar_A(
        biome, 12345, 35, 50, profile=profile
    )

    for layer in LAYERS:
        np.testing.assert_array_equal(
            first[layer],
            second[layer],
        )


@pytest.mark.parametrize("biome", BIOMES)
@pytest.mark.parametrize("complexity", COMPLEXITIES)
def test_generacion_con_perfil_produce_capas_validas(
        biome, complexity
):
    profile = get_complexity_profile(biome, complexity)

    result = generar_A(
        biome, 12345, 35, 50, profile=profile
    )

    for layer in LAYERS:
        assert result[layer].shape == (35, 50)

    assert np.isin(result["collision"], [0, 1]).all()
