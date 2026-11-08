"""
HU-111: pruebas de contrato y regresión de generadores Rhyzome V1.
"""
import pytest
from application.generators.perlin_noise import generate_perlin_map
from application.generators.cellular_automata import generate_map


def generate(algorithm, width=50, height=35, seed=77, **kwargs):
    if algorithm == "perlin":
        return generate_perlin_map(width, height, seed=seed, **kwargs)
    return generate_map(width, height, seed=seed, **kwargs)


@pytest.mark.parametrize("algorithm", ["perlin", "cellular"])
@pytest.mark.parametrize("width,height", [(10, 10), (50, 35), (100, 70)])
def test_dimensions_binary_and_list_contract(algorithm, width, height):
    grid = generate(algorithm, width, height)
    assert isinstance(grid, list)
    assert len(grid) == height
    assert all(isinstance(row, list) and len(row) == width for row in grid)
    assert all(isinstance(v, int) and v in (0, 1) for row in grid for v in row)


@pytest.mark.parametrize("algorithm", ["perlin", "cellular"])
@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_fixed_seed_is_reproducible(algorithm, seed):
    assert generate(algorithm, seed=seed) == generate(algorithm, seed=seed)


@pytest.mark.parametrize("iterations", [0, 1, 5, 10])
def test_cellular_border_is_zero(iterations):
    grid = generate_map(50, 35, iteraciones=iterations, seed=123)
    assert all(v == 0 for v in grid[0] + grid[-1])
    assert all(row[0] == 0 and row[-1] == 0 for row in grid)


def test_perlin_scale_is_ignored_in_v1_regression():
    """Documenta un defecto/limitación conocida; NO lo considera correcto para V2."""
    a = generate_perlin_map(50, 35, scale=0.035, seed=42)
    b = generate_perlin_map(50, 35, scale=0.12, seed=42)
    assert a == b


@pytest.mark.parametrize("algorithm", ["perlin", "cellular"])
def test_none_seed_matches_default_seed(algorithm):
    assert generate(algorithm, seed=None) == generate(algorithm, seed=42)


@pytest.mark.parametrize("algorithm", ["perlin", "cellular"])
def test_seed_change_changes_generated_map(algorithm):
    assert generate(algorithm, seed=11) != generate(algorithm, seed=22)


@pytest.mark.parametrize("threshold,expected", [(0.0, 1), (1.0, 0)])
def test_perlin_threshold_extremes(threshold, expected):
    grid = generate_perlin_map(50, 35, threshold=threshold, seed=42)
    assert all(v == expected for row in grid for v in row)


def test_cellular_probability_one_without_iterations_is_zero():
    grid = generate_map(50, 35, probabilidad=1.0, iteraciones=0, seed=42)
    assert all(v == 0 for row in grid for v in row)


def test_cellular_probability_zero_without_iterations_is_one_inside():
    grid = generate_map(50, 35, probabilidad=0.0, iteraciones=0, seed=42)
    assert all(grid[y][x] == 1 for y in range(1, 34) for x in range(1, 49))
