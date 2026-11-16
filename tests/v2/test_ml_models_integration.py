
import numpy as np
import pytest

from application.v2.services.ml_adapter import (
    predict_v2_complexity,
)


@pytest.mark.parametrize(
    "generator",
    ["perlin", "cellular"],
)
def test_real_model_prediction(generator):
    rng = np.random.default_rng(12345)

    # 0 = transitable, 1 = obstáculo
    collision = rng.choice(
        [0, 1],
        size=(35, 50),
        p=[0.65, 0.35],
    ).tolist()

    result = predict_v2_complexity(
        {"collision": collision},
        generator,
    )

    assert result["model_generator"] == generator

    assert result["predicted_complexity"] in (
        "bajo",
        "medio",
        "alto",
    )

    assert isinstance(result["ml_features"], dict)

    assert result["ml_features"]

    if result["confidence"] is not None:
        assert 0 <= result["confidence"] <= 1
