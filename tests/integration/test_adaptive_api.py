
from unittest.mock import Mock

import pytest

from application.services import adaptive_generator_ml as ml_module


@pytest.fixture
def mock_adaptive(monkeypatch):
    """
    Simula la generación adaptativa sin cargar
    modelos reales ni generar mapas costosos.
    """
    result = {
        "algorithm": "perlin",
        "map": [[1, 0], [0, 1]],
        "seed": 42,
        "predicted_complexity": "medio",
        "requested_complexity": "medio",
        "matches_request": True,
        "confidence": 0.95,
        "target_probability": 0.90,
        "selected_model": "mock_model",
        "model_metrics": {"f1": 0.91},
        "model_artifact": "mock_model.joblib",
        "metrics": {"quality": 85.0},
        "ml_features": {"floor_ratio": 0.5},
        "generated_parameters": {"threshold": 0.5},
        "attempts": 3,
    }

    generator = Mock(return_value=result)

    monkeypatch.setattr(
        ml_module,
        "generate_adaptive_map_ml",
        generator,
    )

    return generator


@pytest.mark.parametrize(
    "algorithm,terrain",
    [
        ("perlin", "bosque"),
        ("perlin", "desierto"),
        ("perlin", "ciudad"),
        ("cellular", "mazmorra"),
    ],
)
def test_adaptive_generate_success(
    client, mock_adaptive, algorithm, terrain
):
    response = client.post(
        "/api/maps/generate-adaptive",
        json={
            "algorithm": algorithm,
            "terrain_type": terrain,
            "complexity": "medium",
            "width": 20,
            "height": 15,
            "seed": 42,
            "attempts": 3,
        },
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["ui_config"]["terrain_type"] == terrain
    assert data["selected_model"] == "mock_model"
    assert "map" in data
    assert "metrics" in data

    mock_adaptive.assert_called_once()

    args = mock_adaptive.call_args.args

    assert args[0] == algorithm
    assert args[1]["width"] == 20
    assert args[1]["height"] == 15
    assert args[2] == 3
    assert args[3] == "medio"


@pytest.mark.parametrize(
    "complexity,expected",
    [
        ("low", "bajo"),
        ("medium", "medio"),
        ("high", "alto"),
    ],
)
def test_adaptive_complexity_mapping(
    client, mock_adaptive, complexity, expected
):
    response = client.post(
        "/api/maps/generate-adaptive",
        json={
            "terrain_type": "bosque",
            "complexity": complexity,
        },
    )

    assert response.status_code == 200

    args = mock_adaptive.call_args.args

    assert args[3] == expected


@pytest.mark.parametrize(
    "terrain,expected_algorithm",
    [
        ("bosque", "perlin"),
        ("desierto", "perlin"),
        ("ciudad", "perlin"),
        ("mazmorra", "cellular"),
    ],
)
def test_adaptive_algorithm_default(
    client, mock_adaptive, terrain, expected_algorithm
):
    response = client.post(
        "/api/maps/generate-adaptive",
        json={"terrain_type": terrain},
    )

    assert response.status_code == 200

    args = mock_adaptive.call_args.args

    assert args[0] == expected_algorithm


@pytest.mark.parametrize(
    "payload",
    [
        {"width": 0},
        {"height": 300},
        {"attempts": 0},
        {"attempts": 51},
        {"algorithm": "invalid"},
        {"terrain_type": "oceano"},
        {"complexity": "extreme"},
        {"tile_size": 2},
        {"algorithm": "perlin", "octaves": 0},
        {"algorithm": "perlin", "threshold": 1.5},
        {"algorithm": "cellular", "probability": 1.5},
        {"algorithm": "cellular", "iterations": 21},
    ],
)
def test_adaptive_invalid_parameters(
    client, mock_adaptive, payload
):
    response = client.post(
        "/api/maps/generate-adaptive",
        json=payload,
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"

    mock_adaptive.assert_not_called()


def test_adaptive_missing_model(client, mock_adaptive):
    mock_adaptive.side_effect = FileNotFoundError(
        "best_perlin.joblib no encontrado"
    )

    response = client.post(
        "/api/maps/generate-adaptive",
        json={
            "terrain_type": "bosque",
            "complexity": "medium",
        },
    )

    data = response.get_json()

    assert response.status_code == 503
    assert data["status"] == "error"
    assert "Modelo ML no disponible" in data["message"]


def test_adaptive_import_error(client, mock_adaptive):
    mock_adaptive.side_effect = ImportError(
        "No se pudo importar el modelo"
    )

    response = client.post(
        "/api/maps/generate-adaptive",
        json={"terrain_type": "bosque"},
    )

    data = response.get_json()

    assert response.status_code == 503
    assert data["status"] == "error"


def test_adaptive_invalid_attempts_type(
    client, mock_adaptive
):
    response = client.post(
        "/api/maps/generate-adaptive",
        json={"attempts": "abc"},
    )

    assert response.status_code == 400
    assert mock_adaptive.call_count == 0
