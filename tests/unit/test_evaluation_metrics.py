
from application.services.evaluation_metrics import evaluar_mapa


def test_evaluar_mapa_returns_expected_metrics():
    grid = [
        [0, 1],
        [0, 0],
    ]

    result = evaluar_mapa(grid)

    expected_keys = {
        "quality",
        "accessibility",
        "connectivity",
        "obstacle_density",
        "connected_components",
        "largest_open_area",
        "border_obstacle_ratio",
        "balance",
    }

    assert set(result.keys()) == expected_keys


def test_evaluar_mapa_returns_valid_quality():
    grid = [
        [0, 0],
        [0, 1],
    ]

    result = evaluar_mapa(grid)

    assert isinstance(result["quality"], float)
    assert 0 <= result["quality"] <= 100


def test_evaluar_mapa_accessibility():
    grid = [
        [0, 1],
        [0, 0],
    ]

    result = evaluar_mapa(grid)

    assert result["accessibility"] == 0.75
    assert result["obstacle_density"] == 0.25

