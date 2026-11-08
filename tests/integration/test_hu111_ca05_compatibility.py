"""
HU-111 / CA-05: contrato de mapas y métricas del backend (etapa 1).
"""
import pytest
from flask import Flask
from importlib import import_module

from application.routes.map_routes import map_bp
from application.services.evaluation_metrics import evaluar_mapa


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(map_bp, url_prefix="/api/maps")
    return app.test_client()


TERRAINS = [
    ("bosque", "perlin"),
    ("desierto", "perlin"),
    ("ciudad", "perlin"),
    ("mazmorra", "cellular"),
]


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_generate_returns_frontend_compatible_contract(client, terrain, algorithm):
    response = client.post("/api/maps/generate", json={
        "terrain_type": terrain,
        "width": 50,
        "height": 35,
        "seed": 42,
        "complexity": "medium",
        "tile_size": 16,
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["algorithm"] == algorithm
    assert body["dimensions"] == {"width": 50, "height": 35}
    assert body["ui_config"] == {
        "terrain_type": terrain, "complexity": "medium", "tile_size": 16
    }
    assert isinstance(body["parameters"], dict)
    assert body["seed"] == 42
    assert isinstance(body["metrics"], dict)
    assert isinstance(body["map"], list)


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_generated_grid_is_rectangular_and_binary(client, terrain, algorithm):
    response = client.post("/api/maps/generate", json={
        "terrain_type": terrain, "width": 32, "height": 24, "seed": 42
    })
    assert response.status_code == 200
    grid = response.get_json()["map"]
    assert len(grid) == 24
    assert all(isinstance(row, list) and len(row) == 32 for row in grid)
    assert all(type(cell) is int and cell in (0, 1) for row in grid for cell in row)


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_generated_map_can_be_evaluated_without_changing_metrics(client, terrain, algorithm):
    generated = client.post("/api/maps/generate", json={
        "terrain_type": terrain, "width": 32, "height": 24, "seed": 42
    })
    assert generated.status_code == 200
    body = generated.get_json()
    evaluated = client.post("/api/maps/evaluate", json={"map": body["map"]})
    assert evaluated.status_code == 200
    result = evaluated.get_json()
    assert result["status"] == "success"
    assert result["metrics"] == body["metrics"]
    assert result["metrics"] == evaluar_mapa(body["map"])


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_metric_values_match_binary_semantics(client, terrain, algorithm):
    generated = client.post("/api/maps/generate", json={
        "terrain_type": terrain, "width": 32, "height": 24, "seed": 42
    })
    assert generated.status_code == 200
    body = generated.get_json()
    grid = body["map"]
    metrics = body["metrics"]
    floor_ratio = sum(cell == 1 for row in grid for cell in row) / (32 * 24)
    assert metrics["accessibility"] == pytest.approx(floor_ratio, abs=0.00005)
    assert metrics["obstacle_density"] == pytest.approx(1 - floor_ratio, abs=0.00005)
    assert 0 <= metrics["connectivity"] <= 1
    assert metrics["connected_components"] >= 0
    assert 0 <= metrics["quality"] <= 100


@pytest.mark.parametrize("grid", [
    [], [[0, 1], [1]], [[0, 2], [1, 0]], [[0, 1], "01"]
])
def test_evaluate_rejects_invalid_grid(client, grid):
    response = client.post("/api/maps/evaluate", json={"map": grid})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"

# ============================================================
# CA-05 / Etapa 2: contrato de /generate-adaptive
# Se simula el servicio ML, pero se calculan métricas reales.
# ============================================================

@pytest.fixture
def adaptive_stub(monkeypatch):
    adaptive_service = import_module(
        "application.services.adaptive_generator_ml"
    )

    def fake_generate_adaptive_map_ml(generator, config, attempts=10,
                                      requested_complexity="medio"):
        width = config["width"]
        height = config["height"]
        # Patrón binario determinista; no requiere cargar modelos ML.
        grid = [[int((x + y) % 3 != 0) for x in range(width)]
                for y in range(height)]
        generated_parameters = (
            {"octaves": 4, "persistence": 0.5, "threshold": 0.5}
            if generator == "perlin"
            else {"probability": 0.45, "iterations": 4,
                  "birth_limit": 4, "death_limit": 3}
        )
        return {
            "map": grid,
            "seed": 123456,
            "algorithm": generator,
            "requested_complexity": requested_complexity,
            "predicted_complexity": requested_complexity,
            "matches_request": True,
            "confidence": 0.95,
            "target_probability": 0.95,
            "metrics": evaluar_mapa(grid),
            "generated_parameters": generated_parameters,
            "parameters": config,
            "attempts": attempts,
            "selected_model": "modelo_simulado_ca05",
            "model_metrics": {},
            "model_artifact": "simulado",
        }

    monkeypatch.setattr(
        adaptive_service,
        "generate_adaptive_map_ml",
        fake_generate_adaptive_map_ml
    )


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_adaptive_contract_and_binary_grid(client, adaptive_stub, terrain, algorithm):
    response = client.post("/api/maps/generate-adaptive", json={
        "terrain_type": terrain,
        "algorithm": algorithm,
        "width": 50,
        "height": 35,
        "seed": 539989,
        "complexity": "high",
        "tile_size": 16,
        "attempts": 10,
    })
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["status"] == "success"
    assert body["algorithm"] == algorithm
    assert body["attempts"] == 10
    assert isinstance(body["seed"], int)
    assert isinstance(body["parameters"], dict)
    assert isinstance(body["generated_parameters"], dict)
    assert isinstance(body["ui_config"], dict)
    assert isinstance(body["metrics"], dict)
    assert len(body["map"]) == 35
    assert all(len(row) == 50 for row in body["map"])
    assert all(type(cell) is int and cell in (0, 1)
               for row in body["map"] for cell in row)
    for metric in ("quality", "accessibility", "connectivity"):
        assert isinstance(body["metrics"][metric], (int, float))
    assert body["metrics"] == evaluar_mapa(body["map"])


@pytest.mark.parametrize("terrain,algorithm", TERRAINS)
def test_adaptive_grid_can_be_evaluated(client, adaptive_stub, terrain, algorithm):
    response = client.post("/api/maps/generate-adaptive", json={
        "terrain_type": terrain,
        "algorithm": algorithm,
        "width": 32,
        "height": 24,
        "complexity": "high",
        "attempts": 3,
    })
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    evaluation = client.post("/api/maps/evaluate", json={"map": body["map"]})
    assert evaluation.status_code == 200, evaluation.get_json()
    assert evaluation.get_json()["metrics"] == body["metrics"]
