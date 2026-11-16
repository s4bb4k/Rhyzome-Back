import numpy as np
import pytest

from application.v2.services.map_service import generate

BIOMES = ("bosque", "desierto", "ciudad", "mazmorra")
COMPLEXITIES = ("bajo", "medio", "alto")
VARIANTS = ("A", "B")


@pytest.mark.parametrize("biome", BIOMES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_sin_complejidad_conserva_resultado_original(
        biome, variant
):
    base = {
        "terrain_type": biome,
        "seed": 12345,
        "width": 50,
        "height": 35,
        "variant": variant,
    }

    original = generate(base)

    assert "generation_profile" not in original

    medio = generate({
        **base,
        "complexity": "medio",
    })

    for layer in ("terrain", "objects", "collision"):
        assert (
                original["layers"][layer]
                == medio["layers"][layer]
        )


@pytest.mark.parametrize("biome", BIOMES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("complexity", COMPLEXITIES)
def test_generacion_con_perfil(
        biome, variant, complexity
):
    result = generate({
        "terrain_type": biome,
        "seed": 12345,
        "width": 50,
        "height": 35,
        "variant": variant,
        "complexity": complexity,
    })

    assert result["status"] == "success"
    assert result["generation_profile"][
               "complexity_requested"
           ] == complexity

    assert result["generation_profile"]["validated"] is False

    collision = np.asarray(
        result["layers"]["collision"]
    )

    assert collision.shape == (35, 50)
    assert np.isin(collision, [0, 1]).all()


@pytest.mark.parametrize(
    "complexity",
    ["extremo", "", 3, True],
)
def test_complejidad_invalida(complexity):
    with pytest.raises(ValueError):
        generate({
            "terrain_type": "bosque",
            "seed": 12345,
            "complexity": complexity,
        })


def test_complejidad_es_opcional():
    result = generate({
        "terrain_type": "bosque",
        "seed": 12345,
    })

    assert result["status"] == "success"
    assert "generation_profile" not in result
