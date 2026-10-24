
import pytest

from application.generators.perlin_noise import generate_perlin_map


def test_perlin_dimensions():
    grid = generate_perlin_map(20, 15, seed=42)

    assert len(grid) == 15
    assert all(len(row) == 20 for row in grid)


def test_perlin_binary_values():
    grid = generate_perlin_map(20, 20, seed=42)

    assert all(cell in (0, 1) for row in grid for cell in row)


def test_perlin_reproducibility():
    first = generate_perlin_map(20, 20, seed=42)
    second = generate_perlin_map(20, 20, seed=42)

    assert first == second


@pytest.mark.parametrize("width,height", [
    (1, 1),
    (5, 10),
    (10, 5),
    (30, 30),
])
def test_perlin_different_dimensions(width, height):
    grid = generate_perlin_map(width, height, seed=123)

    assert len(grid) == height
    assert all(len(row) == width for row in grid)



def test_perlin_threshold_zero():
    grid = generate_perlin_map(10, 10, threshold=0, seed=42)

    # Todos los valores normalizados cumplen noise >= 0.
    assert all(cell == 1 for row in grid for cell in row)


def test_perlin_threshold_one():
    grid = generate_perlin_map(10, 10, threshold=1, seed=42)

    # Con este mapa, ningún valor normalizado alcanza 1.
    assert all(cell == 0 for row in grid for cell in row)


