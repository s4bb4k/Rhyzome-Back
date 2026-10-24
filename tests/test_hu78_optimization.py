
"""
HU-78 - Optimización de tiempos de respuesta de Rhyzome.

Pruebas:
- Extracción de características ML.
- Reutilización de características.
- Equivalencia de evaluación de mapas.
- Validación de mapas vacíos.
- Equivalencia entre extracción original y vectorizada.
- Validación de mapas uniformes.

Ejecutar:
    python -m pytest tests/test_hu78_optimization.py -v
"""

import numpy as np
import pytest


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================
# Sustituye estas rutas por los imports que ya te funcionaron.
# _regions y _binary_entropy deben importarse desde el módulo
# donde está definida extract_ml_feature_dict.

from application.services.feature_extractor import (
    extract_ml_feature_dict,
    extract_feature_dict,
    _regions,
    _binary_entropy,
)

from application.services.evaluation_metrics import (
    evaluar_mapa,
)


# ============================================================
# FIXTURE: MAPA DE PRUEBA
# ============================================================

@pytest.fixture
def sample_grid():
    return np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )


# ============================================================
# IMPLEMENTACIÓN ORIGINAL: REFERENCIA
# ============================================================

def extract_ml_feature_dict_original(grid):
    """
    Implementación original sin vectorización.

    Se conserva exclusivamente como referencia para comprobar
    que la optimización no altera las métricas calculadas.
    """

    arr = np.asarray(grid, dtype=np.uint8)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            "El mapa debe ser una matriz 2D no vacía"
        )

    height, width = arr.shape

    floor_ratio = float(arr.mean())

    sizes = _regions(arr)

    n_regions = len(sizes)
    largest = max(sizes, default=0)

    largest_region_ratio = (
        largest / max(int(arr.sum()), 1)
    )

    horizontal = (
        np.mean(arr[:, 1:] != arr[:, :-1])
        if width > 1
        else 0.0
    )

    vertical = (
        np.mean(arr[1:, :] != arr[:-1, :])
        if height > 1
        else 0.0
    )

    edge_density = float(
        (horizontal + vertical) / 2
    )

    row_entropy = float(
        _binary_entropy(arr.mean(axis=1)).mean()
    )

    col_entropy = float(
        _binary_entropy(arr.mean(axis=0)).mean()
    )

    # Cálculo original: ciclo sobre cada celda.
    padded = np.pad(
        arr.astype(float),
        1,
        mode="edge",
    )

    local_vars = [
        np.var(padded[y:y + 3, x:x + 3])
        for y in range(height)
        for x in range(width)
    ]

    local_variance = float(np.mean(local_vars))

    internal_edges = int(
        np.sum(arr[:, 1:] != arr[:, :-1])
        + np.sum(arr[1:, :] != arr[:-1, :])
    )

    border_open = int(
        arr[0, :].sum()
        + arr[-1, :].sum()
        + arr[:, 0].sum()
        + arr[:, -1].sum()
    )

    perimeter_area_ratio = float(
        (internal_edges + border_open)
        / max(int(arr.sum()), 1)
    )

    return {
        "width": int(width),
        "height": int(height),
        "floor_ratio": floor_ratio,
        "n_regions": int(n_regions),
        "largest_region_ratio": float(
            largest_region_ratio
        ),
        "edge_density": edge_density,
        "row_entropy": row_entropy,
        "col_entropy": col_entropy,
        "local_variance": local_variance,
        "perimeter_area_ratio": perimeter_area_ratio,
    }


# ============================================================
# FUNCIÓN AUXILIAR: COMPARACIÓN DE MÉTRICAS
# ============================================================

def assert_features_equivalent(original, optimized):
    """
    Compara las características de ambas implementaciones.

    Los valores enteros deben coincidir exactamente.
    Los valores decimales se comparan con tolerancia.
    """

    assert original.keys() == optimized.keys(), (
        "Las implementaciones devuelven características "
        "diferentes"
    )

    for name, original_value in original.items():

        optimized_value = optimized[name]

        if isinstance(
            original_value,
            (int, np.integer),
        ):
            assert original_value == optimized_value, (
                f"Diferencia en {name}: "
                f"original={original_value}, "
                f"optimizado={optimized_value}"
            )

        else:
            assert np.isclose(
                original_value,
                optimized_value,
                rtol=1e-10,
                atol=1e-12,
            ), (
                f"Diferencia en {name}: "
                f"original={original_value}, "
                f"optimizado={optimized_value}"
            )


# ============================================================
# TEST 1: EXTRACCIÓN ML
# ============================================================

def test_extraccion_ml(sample_grid):

    features = extract_ml_feature_dict(sample_grid)

    assert features["width"] == 5
    assert features["height"] == 5

    assert 0 <= features["floor_ratio"] <= 1

    assert 0 <= features["local_variance"] <= 1


# ============================================================
# TEST 2: REUTILIZACIÓN DE CARACTERÍSTICAS
# ============================================================

def test_reutilizacion_caracteristicas(sample_grid):

    ml_features = extract_ml_feature_dict(sample_grid)

    original = extract_feature_dict(sample_grid)

    optimized = extract_feature_dict(
        sample_grid,
        ml_features=ml_features,
    )

    assert original == optimized


# ============================================================
# TEST 3: EVALUACIÓN EQUIVALENTE
# ============================================================

def test_evaluacion_equivalente(sample_grid):

    ml_features = extract_ml_feature_dict(sample_grid)

    original = evaluar_mapa(sample_grid)

    optimized = evaluar_mapa(
        sample_grid,
        ml_features=ml_features,
    )

    assert original == optimized


# ============================================================
# TEST 4: MAPA VACÍO
# ============================================================

def test_mapa_vacio():

    with pytest.raises(ValueError):
        extract_ml_feature_dict([])


# ============================================================
# TESTS 5-12: EQUIVALENCIA DE VECTORIZACIÓN
# ============================================================

@pytest.mark.parametrize(
    "height,width,seed",
    [
        (5, 5, 100),
        (10, 10, 200),
        (35, 50, 300),
        (50, 50, 400),
        (80, 80, 500),
        (1, 10, 600),
        (10, 1, 700),
        (1, 1, 800),
    ],
)
def test_hu78_vectorizacion_equivalente(
    height,
    width,
    seed,
):
    """
    Comprueba la equivalencia de las características
    en mapas binarios de diferentes dimensiones.
    """

    rng = np.random.default_rng(seed)

    grid = rng.integers(
        0,
        2,
        size=(height, width),
        dtype=np.uint8,
    )

    original = extract_ml_feature_dict_original(grid)

    optimized = extract_ml_feature_dict(grid)

    assert_features_equivalent(
        original,
        optimized,
    )


# ============================================================
# TESTS 13-14: MAPAS UNIFORMES
# ============================================================

@pytest.mark.parametrize(
    "value",
    [0, 1],
)
def test_hu78_mapas_uniformes(value):
    """
    Verifica mapas compuestos únicamente por
    obstáculos o únicamente por celdas transitables.
    """

    grid = np.full(
        (35, 50),
        value,
        dtype=np.uint8,
    )

    original = extract_ml_feature_dict_original(grid)

    optimized = extract_ml_feature_dict(grid)

    assert_features_equivalent(
        original,
        optimized,
    )
