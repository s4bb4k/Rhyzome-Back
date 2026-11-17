"""
HU-111 / CA-03: contrato HTTP y parámetros de /generate.
"""
import pytest
from flask import Flask
from application.routes.map_routes import map_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(map_bp, url_prefix="/api/maps")
    return app.test_client()


def send(client, **payload):
    return client.post("/api/maps/generate", json=payload)


@pytest.mark.parametrize("terrain,algorithm", [
    ("bosque", "perlin"), ("desierto", "perlin"),
    ("ciudad", "perlin"), ("mazmorra", "cellular"),
])
@pytest.mark.parametrize("width,height", [(10, 10), (50, 35), (100, 70)])
def test_valid_dimensions_and_terrain(client, terrain, algorithm, width, height):
    response = send(client, terrain_type=terrain, width=width, height=height, seed=42)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["algorithm"] == algorithm
    assert body["dimensions"] == {"width": width, "height": height}
    assert len(body["map"]) == height
    assert all(len(row) == width for row in body["map"])
    assert all(value in (0, 1) for row in body["map"] for value in row)


@pytest.mark.parametrize("field,value", [
    ("width", 9), ("width", 201), ("height", 9), ("height", 201),
    ("width", "abc"), ("height", None),
    ("seed", -1), ("seed", 2147483648), ("seed", "abc"),
    ("tile_size", 3), ("tile_size", 65),
    ("terrain_type", "oceano"), ("complexity", "extrema"),
    ("algorithm", "unknown"),
])
def test_invalid_common_parameters_return_400(client, field, value):
    response = send(client, **{field: value})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
    assert response.get_json()["message"]


@pytest.mark.parametrize("field,value", [
    ("scale", 0.004), ("scale", 1.01),
    ("octaves", 0), ("octaves", 9),
    ("persistence", 0.09), ("persistence", 1.01),
    ("threshold", -0.01), ("threshold", 1.01),
    ("octaves", "incorrecto"), ("threshold", "incorrecto"),
])
def test_invalid_perlin_parameters_return_400(client, field, value):
    response = send(client, terrain_type="bosque", **{field: value})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


@pytest.mark.parametrize("field,value", [
    ("probability", -0.01), ("probability", 1.01),
    ("iterations", -1), ("iterations", 21),
    ("birth_limit", 0), ("birth_limit", 9),
    ("death_limit", 0), ("death_limit", 9),
    ("iterations", "incorrecto"), ("probability", "incorrecto"),
])
def test_invalid_cellular_parameters_return_400(client, field, value):
    response = send(client, terrain_type="mazmorra", **{field: value})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


@pytest.mark.parametrize("parameters", [
    {"octaves": 1, "persistence": 0.1, "threshold": 0.0, "scale": 0.005},
    {"octaves": 8, "persistence": 1.0, "threshold": 1.0, "scale": 1.0},
])
def test_valid_perlin_parameter_bounds(client, parameters):
    response = send(client, terrain_type="bosque", width=50, height=35, seed=42, **parameters)
    assert response.status_code == 200
    assert response.get_json()["parameters"] == parameters


@pytest.mark.parametrize("parameters", [
    {"probability": 0.0, "iterations": 0, "birth_limit": 1, "death_limit": 1},
    {"probability": 1.0, "iterations": 20, "birth_limit": 8, "death_limit": 8},
])
def test_valid_cellular_parameter_bounds(client, parameters):
    response = send(client, terrain_type="mazmorra", width=50, height=35, seed=42, **parameters)
    assert response.status_code == 200
    assert response.get_json()["parameters"] == parameters


@pytest.mark.parametrize("field,value", [
    ("width", 10.5),
    ("iterations", 2.5),
    ("octaves", 3.5)
])
def test_fractional_integer_inputs_should_be_rejected(client, field, value):
    terrain = "mazmorra" if field == "iterations" else "bosque"
    response = send(client, terrain_type=terrain, **{field: value})
    assert response.status_code == 400


def test_nan_threshold_should_be_rejected(client):
    response = send(client, terrain_type="bosque", threshold="nan")
    assert response.status_code == 400