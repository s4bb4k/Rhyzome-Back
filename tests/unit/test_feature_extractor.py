import pytest

from application.services.feature_extractor import (
    extract_feature_dict,
    extract_features,
    FEATURE_NAMES,
)



def test_completely_open_map():
    grid = [
        [1, 1],
        [1, 1],
    ]

    features = extract_feature_dict(grid)

    assert features["accessibility"] == 1.0
    assert features["obstacle_density"] == 0.0
    assert features["connectivity"] == 1.0
    assert features["connected_components"] == 1


def test_completely_blocked_map():
    grid = [
        [0, 0],
        [0, 0],
    ]

    features = extract_feature_dict(grid)

    assert features["accessibility"] == 0.0
    assert features["obstacle_density"] == 1.0
    assert features["connectivity"] == 0.0
    assert features["connected_components"] == 0



def test_disconnected_open_areas():
    grid = [
        [0, 1],
        [1, 0],
    ]

    features = extract_feature_dict(grid)

    assert features["connected_components"] == 2
    assert features["connectivity"] == 0.5
    assert features["largest_open_area"] == 0.25


def test_feature_vector_has_expected_length():
    grid = [
        [0, 1],
        [0, 0],
    ]

    features = extract_features(grid)

    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(value, (int, float)) for value in features)
