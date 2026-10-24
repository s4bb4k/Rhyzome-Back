
from unittest.mock import Mock

import pytest

from application.services import adaptive_generator_ml as module


def test_load_model_missing(tmp_path, monkeypatch):
    module._load_model.cache_clear()
    monkeypatch.setattr(module, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="No existe"):
        module._load_model("cellular")


def test_load_model_existing(tmp_path, monkeypatch):
    module._load_model.cache_clear()
    monkeypatch.setattr(module, "MODEL_DIR", tmp_path)

    model_path = tmp_path / "best_cellular.joblib"
    model_path.touch()

    model = Mock()
    loader = Mock(return_value=model)

    monkeypatch.setattr(module.joblib, "load", loader)

    assert module._load_model("cellular") is model
    loader.assert_called_once_with(model_path)


@pytest.mark.parametrize(
    "generator,expected",
    [
        ("cellular", "generate_cellular"),
        ("perlin", "generate_perlin_map"),
    ],
)
def test_candidate_dispatch(monkeypatch, generator, expected):
    cellular = Mock(return_value=[[0]])
    perlin = Mock(return_value=[[1]])

    monkeypatch.setattr(module, "generate_cellular", cellular)
    monkeypatch.setattr(module, "generate_perlin_map", perlin)

    config = {
        "width": 10,
        "height": 12,
        "terrain_type": "bosque",
    }

    grid, parameters = module._candidate(
        generator,
        config,
        seed=42,
        target="medio",
    )

    if generator == "cellular":
        assert grid == [[0]]
        assert set(parameters) == {
            "probability",
            "iterations",
            "birth_limit",
            "death_limit",
        }
        cellular.assert_called_once()
        perlin.assert_not_called()
    else:
        assert grid == [[1]]
        assert set(parameters) == {
            "octaves",
            "persistence",
            "threshold",
        }
        perlin.assert_called_once()
        cellular.assert_not_called()


def setup_adaptive_mocks(monkeypatch, model):
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda generator: model,
    )

    def fake_read_json(name):
        if name == "feature_config.json":
            return {
                "features": ["floor_ratio"],
                "target_encoding": {
                    "bajo": 0,
                    "medio": 1,
                    "alto": 2,
                },
            }

        if name == "experiment_manifest.json":
            return {
                "winners": {
                    "cellular": {
                        "model": "mock_model",
                        "metrics": {"f1": 0.9},
                        "artifact": "best_cellular.joblib",
                    },
                    "perlin": {
                        "model": "mock_model",
                        "metrics": {"f1": 0.9},
                        "artifact": "best_perlin.joblib",
                    },
                },
            }

        raise ValueError(f"Archivo inesperado: {name}")

    monkeypatch.setattr(
        module,
        "_read_json",
        fake_read_json,
    )

    monkeypatch.setattr(
        module,
        "extract_ml_feature_dict",
        lambda grid: {"floor_ratio": 0.5},
    )

    # HU-78: Compatibilidad con reutilización de características ML
    monkeypatch.setattr(
        module,
        "evaluar_mapa",
        lambda grid, ml_features=None: {"quality": 50.0},
    )


@pytest.mark.parametrize("generator", ["cellular", "perlin"])
def test_adaptive_selects_best_prediction(monkeypatch, generator):
    model = Mock()

    model.predict.side_effect = [
        [0],
        [1],
        [2],
    ]

    model.classes_ = [0, 1, 2]

    model.predict_proba.side_effect = [
        [[0.8, 0.1, 0.1]],
        [[0.1, 0.8, 0.1]],
        [[0.1, 0.2, 0.7]],
    ]

    setup_adaptive_mocks(monkeypatch, model)

    grids = [
        ([[0]], {"iteration": 1}),
        ([[1]], {"iteration": 2}),
        ([[0, 1]], {"iteration": 3}),
    ]

    monkeypatch.setattr(
        module,
        "_candidate",
        Mock(side_effect=grids),
    )

    result = module.generate_adaptive_map_ml(
        generator,
        {"seed": 42},
        attempts=3,
        requested_complexity="medio",
    )

    assert result["map"] == [[1]]
    assert result["predicted_complexity"] == "medio"
    assert result["matches_request"] is True
    assert result["algorithm"] == generator
    assert result["selected_model"] == "mock_model"
    assert result["attempts"] == 3
    assert result["target_probability"] == 0.8
    assert model.predict.call_count == 3


def test_adaptive_reproducible_seeds(monkeypatch):
    model = Mock()

    model.predict.return_value = [1]
    model.classes_ = [0, 1, 2]
    model.predict_proba.return_value = [[0.1, 0.8, 0.1]]

    setup_adaptive_mocks(monkeypatch, model)

    seeds = []

    def candidate(generator, config, seed, target):
        seeds.append(seed)
        return [[1]], {"seed": seed}

    monkeypatch.setattr(module, "_candidate", candidate)

    module.generate_adaptive_map_ml(
        "cellular",
        {"seed": 12},
        attempts=3,
    )

    first = seeds[:]
    seeds.clear()

    module.generate_adaptive_map_ml(
        "cellular",
        {"seed": 12},
        attempts=3,
    )

    assert seeds == first
    assert len(seeds) == 3


def test_adaptive_rejects_invalid_complexity(monkeypatch):
    model = Mock()
    setup_adaptive_mocks(monkeypatch, model)

    with pytest.raises(ValueError, match="Complejidad ML"):
        module.generate_adaptive_map_ml(
            "cellular",
            {"seed": 42},
            attempts=3,
            requested_complexity="extremo",
        )

@pytest.fixture(autouse=True)
def limpiar_cache_modelos():
    cache_clear = getattr(module._load_model, "cache_clear", None)

    if callable(cache_clear):
        cache_clear()

    yield

    cache_clear = getattr(module._load_model, "cache_clear", None)

    if callable(cache_clear):
        cache_clear()