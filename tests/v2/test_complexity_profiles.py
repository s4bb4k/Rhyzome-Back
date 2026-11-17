import pytest

from application.v2.services.complexity_profiles import (
    COMPLEXITY_PROFILES,
    get_complexity_profile,
)


@pytest.mark.parametrize(
    "biome",
    ["bosque", "desierto", "ciudad", "mazmorra"],
)
@pytest.mark.parametrize(
    "complexity",
    ["bajo", "medio", "alto"],
)
def test_perfiles_existen(biome, complexity):
    profile = get_complexity_profile(biome, complexity)

    assert isinstance(profile, dict)
    assert len(profile) > 0


def test_perfiles_no_modifican_configuracion_original():
    profile = get_complexity_profile("bosque", "medio")
    profile["tree_probability"] = 0.99

    assert (
        COMPLEXITY_PROFILES["bosque"]["medio"]["tree_probability"]
        == 0.31
    )


def test_bioma_invalido():
    with pytest.raises(ValueError):
        get_complexity_profile("invalido", "medio")


def test_complejidad_invalida():
    with pytest.raises(ValueError):
        get_complexity_profile("bosque", "extremo")


@pytest.mark.parametrize(
    "biome",
    ["bosque", "desierto", "ciudad", "mazmorra"],
)
def test_perfiles_mantienen_parametros_consistentes(biome):
    bajo = get_complexity_profile(biome, "bajo")
    medio = get_complexity_profile(biome, "medio")
    alto = get_complexity_profile(biome, "alto")

    assert bajo.keys() == medio.keys() == alto.keys()
