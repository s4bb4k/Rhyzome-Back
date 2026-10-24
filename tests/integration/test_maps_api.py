
import pytest


@pytest.mark.parametrize(
    "algorithm,terrain",
    [
        ("perlin", "bosque"),
        ("cellular", "mazmorra"),
    ]
)
def test_generate_map_success(client, algorithm, terrain):
    response = client.post(
        "/api/maps/generate",
        json={
            "algorithm": algorithm,
            "terrain_type": terrain,
            "width": 20,
            "height": 15,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    body = response.get_json()

    assert body["status"] == "success"
    assert body["algorithm"] == algorithm
    assert body["dimensions"] == {
        "width": 20,
        "height": 15,
    }
    assert len(body["map"]) == 15
    assert all(len(row) == 20 for row in body["map"])
    assert "metrics" in body


def test_generate_map_reproducible(client):
    payload = {
        "algorithm": "perlin",
        "width": 20,
        "height": 20,
        "seed": 123,
    }

    first = client.post("/api/maps/generate", json=payload)
    second = client.post("/api/maps/generate", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["map"] == second.get_json()["map"]


@pytest.mark.parametrize(
    "payload",
    [
        {"width": 0},
        {"height": 300},
        {"algorithm": "invalido"},
        {"terrain_type": "oceano"},
        {"tile_size": 2},
        {"algorithm": "perlin", "octaves": 0},
        {"algorithm": "cellular", "probability": 1.5},
    ],
)
def test_generate_map_invalid_parameters(client, payload):
    response = client.post(
        "/api/maps/generate",
        json=payload,
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_evaluate_map_success(client):
    response = client.post(
        "/api/maps/evaluate",
        json={
            "map": [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
            ]
        },
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    assert "metrics" in response.get_json()


@pytest.mark.parametrize(
    "invalid_map",
    [
        [],
        [[0, 1], [1]],
        [[0, 2], [1, 0]],
        "mapa",
        None,
    ],
)
def test_evaluate_map_invalid(client, invalid_map):
    response = client.post(
        "/api/maps/evaluate",
        json={"map": invalid_map},
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
