
import numpy as np
import pytest

from application.v2.generators.semantic import generar_A
from application.v2.generators.optimized import generar_B


@pytest.mark.parametrize(
    "bioma",
    ["bosque", "desierto", "ciudad", "mazmorra"],
)
def test_generador_a_es_reproducible(bioma):
    primero = generar_A(bioma, 12345, 35, 50)
    segundo = generar_A(bioma, 12345, 35, 50)

    for capa in ("terrain", "objects", "collision"):
        np.testing.assert_array_equal(
            primero[capa],
            segundo[capa],
        )


@pytest.mark.parametrize(
    "bioma",
    ["bosque", "desierto", "ciudad", "mazmorra"],
)
def test_generador_b_es_reproducible(bioma):
    primero, _ = generar_B(bioma, 12345, 35, 50)
    segundo, _ = generar_B(bioma, 12345, 35, 50)

    for capa in ("terrain", "objects", "collision"):
        np.testing.assert_array_equal(
            primero[capa],
            segundo[capa],
        )


@pytest.mark.parametrize(
    "bioma",
    ["bosque", "desierto", "ciudad", "mazmorra"],
)
def test_generadores_producen_colisiones_binarias(bioma):
    for mapa in (
        generar_A(bioma, 12345, 35, 50),
        generar_B(bioma, 12345, 35, 50)[0],
    ):
        collision = mapa["collision"]

        assert collision.shape == (35, 50)
        assert np.isin(collision, [0, 1]).all()
