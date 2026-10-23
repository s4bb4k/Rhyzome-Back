
import pytest

from application.generators.cellular_automata import generate_map


def test_cellular_dimensions():
    grid = generate_map(20, 15, 0.45, 4, seed=42)

    assert len(grid) == 15
    assert all(len(row) == 20 for row in grid)


def test_cellular_binary_values():
    grid = generate_map(20, 20, 0.45, 4, seed=42)

    assert all(cell in (0, 1) for row in grid for cell in row)


def test_cellular_reproducibility():
    first = generate_map(20, 20, 0.45, 4, seed=42)
    second = generate_map(20, 20, 0.45, 4, seed=42)

    assert first == second


@pytest.mark.parametrize("width,height", [
    (1, 1),
    (5, 10),
    (10, 5),
    (30, 30),
])
def test_cellular_different_dimensions(width, height):
    grid = generate_map(width, height, 0.45, 4, seed=42)

    assert len(grid) == height
    assert all(len(row) == width for row in grid)


def test_cellular_zero_probability_without_iterations():
    grid = generate_map(10, 10, 0.0, 0, seed=42)

    assert all(cell == 0 for row in grid for cell in row)


def test_cellular_full_probability_without_iterations():
    grid = generate_map(10, 10, 1.0, 0, seed=42)

    assert all(cell == 1 for row in grid for cell in row)


def test_cellular_iterations_change_expected_grid():
    initial = generate_map(1, 1, 0.0, 0, seed=42)
    evolved = generate_map(1, 1, 0.0, 1, seed=42)

    assert initial == [[0]]
    assert evolved == [[1]]
