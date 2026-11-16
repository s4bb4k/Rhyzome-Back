from unittest.mock import patch

import pytest

from application.v2.services.adaptive_service import generate_adaptive


def fake_generate(data):
    seed = data["seed"]

    predictions = {
        100: ("bajo", 0.10),
        101: ("medio", 0.75),
        102: ("medio", 0.90),
        103: ("alto", 0.20),
    }

    predicted, target_probability = predictions.get(
        seed, ("bajo", 0.05)
    )

    return {
        "status": "success",
        "version": "v2",
        "seed": seed,
        "terrain_type": data.get("terrain_type", "bosque"),
        "layers": {
            "terrain": [[0]],
            "objects": [[0]],
            "collision": [[0]],
        },
        "metrics": {},
        "ml": {
            "predicted_complexity": predicted,
            "confidence": 0.90,
            "class_probabilities": {
                "bajo": 0.05,
                "medio": target_probability,
                "alto": 0.05,
            },
            "model_generator": "perlin",
        },
    }


@pytest.fixture
def request_data():
    return {
        "terrain_type": "bosque",
        "width": 20,
        "height": 20,
        "seed": 100,
        "variant": "B",
        "complexity": "medio",
        "attempts": 4,
    }


@patch(
    "application.v2.services.adaptive_service.generate",
    side_effect=fake_generate,
)
def test_selects_best_candidate(mock_generate, request_data):
    result = generate_adaptive(request_data)

    assert result["seed"] == 102
    assert result["adaptive"]["selected_attempt"] == 3
    assert result["adaptive"]["target_matched"] is True
    assert result["adaptive"]["exact_matches"] == 2
    assert result["adaptive"]["attempts_executed"] == 4
    assert mock_generate.call_count == 4


@patch(
    "application.v2.services.adaptive_service.generate",
    side_effect=fake_generate,
)
def test_preserves_semantic_layers(mock_generate, request_data):
    result = generate_adaptive(request_data)

    assert set(result["layers"]) == {
        "terrain",
        "objects",
        "collision",
    }
    assert result["status"] == "success"
    assert result["adaptive"]["compatibility_validated"] is False


@patch(
    "application.v2.services.adaptive_service.generate",
    side_effect=fake_generate,
)
def test_no_exact_match(mock_generate, request_data):
    request_data["seed"] = 200
    request_data["attempts"] = 3

    result = generate_adaptive(request_data)

    assert result["adaptive"]["target_matched"] is False
    assert result["adaptive"]["exact_matches"] == 0
    assert result["adaptive"]["attempts_executed"] == 3


@pytest.mark.parametrize("value", [
    "muy_alto",
    "",
    None,
    3,
])
def test_invalid_complexity(value, request_data):
    request_data["complexity"] = value

    with pytest.raises(ValueError):
        generate_adaptive(request_data)


@pytest.mark.parametrize("value", [
    0,
    51,
    True,
    "abc",
    2.5,
])
def test_invalid_attempts(value, request_data):
    request_data["attempts"] = value

    with pytest.raises(ValueError):
        generate_adaptive(request_data)


def test_invalid_include_ml(request_data):
    request_data["include_ml"] = False

    with pytest.raises(ValueError):
        generate_adaptive(request_data)


def test_invalid_seed_overflow(request_data):
    request_data["seed"] = 2147483647
    request_data["attempts"] = 2

    with pytest.raises(ValueError):
        generate_adaptive(request_data)
