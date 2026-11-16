
import numpy as np
import pytest

from application.v2.services.ml_adapter import (
    collision_to_ml_grid,
    predict_v2_complexity,
)


def test_collision_conversion():
    collision = [
        [0, 1, 0],
        [1, 0, 1],
    ]

    result = collision_to_ml_grid(collision)

    expected = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_invalid_generator():
    with pytest.raises(ValueError):
        predict_v2_complexity(
            {"collision": [[0, 1]]},
            "invalid",
        )


def test_invalid_collision():
    with pytest.raises(ValueError):
        collision_to_ml_grid([
            [0, 2],
            [1, 0],
        ])


def test_empty_collision():
    with pytest.raises(ValueError):
        collision_to_ml_grid([])
