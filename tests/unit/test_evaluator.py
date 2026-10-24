import pytest

from application.services.evaluator import evaluate


def test_evaluate_returns_score_in_range():
    features = {
        "connectivity": 0.8,
        "accessibility": 0.7,
        "obstacle_density": 0.3,
        "connected_components": 1,
        "largest_open_area": 0.7,
        "border_obstacle_ratio": 0.5,
        "balance": 0.9,
    }

    score = evaluate(features)

    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_evaluate_accepts_feature_list():
    features = [0.8, 0.7, 0.3, 1, 0.7, 0.5, 0.9]

    score = evaluate(features)

    assert 0 <= score <= 100


def test_evaluate_clamps_high_score():
    features = {
        "connectivity": 10,
        "obstacle_density": 0.35,
        "connected_components": 0,
        "largest_open_area": 10,
        "border_obstacle_ratio": 10,
        "balance": 10,
    }

    assert evaluate(features) == 100.0


def test_evaluate_is_deterministic():
    features = [0.8, 0.7, 0.3, 1, 0.7, 0.5, 0.9]

    assert evaluate(features) == evaluate(features)
