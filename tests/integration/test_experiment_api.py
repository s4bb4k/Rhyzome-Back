
import csv
import json

import pytest

from application.routes import experiment_routes as module


@pytest.fixture
def experiment_data(tmp_path, monkeypatch):
    """
    Crea un experimento temporal para las pruebas.

    No modifica los archivos reales de Rhyzome.
    """
    experiment_dir = tmp_path / "experiment"
    artifact_dir = experiment_dir / "artifacts"
    dataset_dir = experiment_dir / "dataset"
    maps_dir = dataset_dir / "maps"

    artifact_dir.mkdir(parents=True)
    maps_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module, "EXPERIMENT_DIR", experiment_dir
    )
    monkeypatch.setattr(
        module, "ARTIFACT_DIR", artifact_dir
    )

    manifest = {
        "dataset": {"total_maps": 10000},
        "validation": {"method": "cross_validation"},
        "winners": {
            "cellular": {"model": "RandomForest"},
            "perlin": {"model": "RandomForest"},
        },
        "generated_at_utc": "2026-10-01T00:00:00Z",
        "generated_by": "pytest",
        "versions": {"python": "3.14"},
        "artifacts_sha256": {},
    }

    results = {
        "cellular": {"accuracy": 0.9},
        "perlin": {"accuracy": 0.9},
    }

    features = {
        "features": ["width", "height", "floor_ratio"],
        "target_encoding": {
            "bajo": 0,
            "medio": 1,
            "alto": 2,
        },
    }

    for name, content in [
        ("experiment_manifest.json", manifest),
        ("experiment_results.json", results),
        ("feature_config.json", features),
    ]:
        path = artifact_dir / name
        path.write_text(
            json.dumps(content),
            encoding="utf-8",
        )

    rows = [
        {
            "map_id": "cellular-001",
            "generator": "cellular",
            "width": "20",
            "height": "20",
        },
        {
            "map_id": "perlin-001",
            "generator": "perlin",
            "width": "30",
            "height": "30",
        },
        {
            "map_id": "cellular-002",
            "generator": "cellular",
            "width": "25",
            "height": "25",
        },
    ]

    index_path = dataset_dir / "dataset_index.csv"

    with index_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "map_id",
                "generator",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        generator_dir = maps_dir / row["generator"]
        generator_dir.mkdir(exist_ok=True)

        map_path = generator_dir / (
            row["map_id"] + ".json"
        )

        map_path.write_text(
            json.dumps({
                "map_id": row["map_id"],
                "generator": row["generator"],
                "map": [[0, 1], [1, 0]],
            }),
            encoding="utf-8",
        )

    return {
        "experiment_dir": experiment_dir,
        "artifact_dir": artifact_dir,
        "dataset_dir": dataset_dir,
    }


def test_experiment_summary_success(client, experiment_data):
    response = client.get("/api/experiment")

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["dataset"]["total_maps"] == 10000
    assert "validation" in data
    assert "winners" in data
    assert "features" in data
    assert "experiment" in data
    assert "provenance" in data


def test_experiment_summary_missing_files(
    client, experiment_data
):
    manifest = (
        experiment_data["artifact_dir"]
        / "experiment_manifest.json"
    )
    manifest.unlink()

    response = client.get("/api/experiment")

    data = response.get_json()

    assert response.status_code == 404
    assert data["status"] == "error"
    assert "experimento" in data["message"]


def test_experiment_list_maps_success(
    client, experiment_data
):
    response = client.get("/api/experiment/maps")

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["total"] == 3
    assert len(data["maps"]) == 3
    assert data["limit"] == 20
    assert data["offset"] == 0


@pytest.mark.parametrize(
    "generator,expected_count",
    [
        ("cellular", 2),
        ("perlin", 1),
    ],
)
def test_experiment_list_maps_filter(
    client, experiment_data, generator, expected_count
):
    response = client.get(
        "/api/experiment/maps",
        query_string={"generator": generator},
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["total"] == expected_count
    assert len(data["maps"]) == expected_count
    assert all(
        row["generator"] == generator
        for row in data["maps"]
    )


def test_experiment_list_maps_pagination(
    client, experiment_data
):
    response = client.get(
        "/api/experiment/maps",
        query_string={
            "limit": 1,
            "offset": 1,
        },
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["total"] == 3
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert len(data["maps"]) == 1
    assert data["maps"][0]["map_id"] == "perlin-001"


@pytest.mark.parametrize(
    "generator",
    ["invalid", "random", "forest"],
)
def test_experiment_list_maps_invalid_generator(
    client, experiment_data, generator
):
    response = client.get(
        "/api/experiment/maps",
        query_string={"generator": generator},
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


@pytest.mark.parametrize(
    "query",
    [
        {"limit": "abc"},
        {"offset": "xyz"},
        {"limit": "1.5"},
    ],
)
def test_experiment_list_maps_invalid_pagination(
    client, experiment_data, query
):
    response = client.get(
        "/api/experiment/maps",
        query_string=query,
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_experiment_list_maps_limits(
    client, experiment_data
):
    response = client.get(
        "/api/experiment/maps",
        query_string={
            "limit": 999,
            "offset": -10,
        },
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["limit"] == 100
    assert data["offset"] == 0


def test_experiment_list_maps_missing_index(
    client, experiment_data
):
    index_path = (
        experiment_data["dataset_dir"]
        / "dataset_index.csv"
    )
    index_path.unlink()

    response = client.get("/api/experiment/maps")

    assert response.status_code == 404
    assert response.get_json()["status"] == "error"


@pytest.mark.parametrize(
    "map_id",
    [
        "cellular-001",
        "perlin-001",
    ],
)
def test_experiment_get_map_success(
    client, experiment_data, map_id
):
    response = client.get(
        f"/api/experiment/maps/{map_id}"
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["training_map"]["map_id"] == map_id
    assert "note" in data


@pytest.mark.parametrize(
    "map_id",
    [
        "invalid-001",
        "forest-001",
        "cellular-abc_123",
    ],
)
def test_experiment_get_map_invalid_id(
    client, experiment_data, map_id
):
    response = client.get(
        f"/api/experiment/maps/{map_id}"
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_experiment_get_map_not_found(
    client, experiment_data
):
    response = client.get(
        "/api/experiment/maps/cellular-999"
    )

    data = response.get_json()

    assert response.status_code == 404
    assert data["status"] == "error"
    assert data["message"] == "Mapa no encontrado"
