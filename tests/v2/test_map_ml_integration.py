
from unittest.mock import patch

import pytest

from application.v2.services.map_service import generate


@pytest.fixture
def payload():
    return {
        "terrain_type": "bosque",
        "width": 20,
        "height": 20,
        "seed": 12345,
        "variant": "B",
    }


def test_generation_without_ml(payload):
    result = generate(payload)

    assert result["status"] == "success"
    assert "ml" not in result

    assert set(result["layers"]) == {
        "terrain",
        "objects",
        "collision",
    }


def test_generation_with_ml(payload):
    payload["include_ml"] = True

    with patch(
        "application.v2.services.map_service.predict_v2_complexity"
    ) as mock_predict:
        mock_predict.return_value = {
            "predicted_complexity": "medio",
            "confidence": 0.85,
            "class_probabilities": {},
            "ml_features": {},
            "model_generator": "perlin",
        }

        result = generate(payload)

        assert result["status"] == "success"
        assert result["ml"]["predicted_complexity"] == "medio"
        assert result["ml"]["compatibility_validated"] is False

        mock_predict.assert_called_once_with(
            result["layers"],
            "perlin",
        )


def test_invalid_include_ml(payload):
    payload["include_ml"] = "true"

    with pytest.raises(ValueError):
        generate(payload)


def test_dungeon_selects_cellular(payload):
    payload["terrain_type"] = "mazmorra"
    payload["include_ml"] = True

    with patch(
        "application.v2.services.map_service.predict_v2_complexity"
    ) as mock_predict:
        mock_predict.return_value = {
            "predicted_complexity": "alto",
            "confidence": 0.8,
            "class_probabilities": {},
            "ml_features": {},
            "model_generator": "cellular",
        }

        result = generate(payload)

        mock_predict.assert_called_once_with(
            result["layers"],
            "cellular",
        )
