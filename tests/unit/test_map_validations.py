import pytest
from application.routes.map_routes import _integer, _number, _frontend_config, _algorithm


@pytest.mark.parametrize('value,expected', [('12', 12), (12, 12), (12.0, 12)])
def test_integer_valid(value, expected):
    assert _integer({'width': value}, 'width', 10, 1, 20) == expected


@pytest.mark.parametrize('value', ['abc', None, 0, 21])
def test_integer_invalid(value):
    with pytest.raises(ValueError):
        _integer({'width': value}, 'width', 10, 1, 20)


def test_integer_default():
    assert _integer({}, 'width', 10, 1, 20) == 10


@pytest.mark.parametrize('value', ['0.45', 0.45, 0, 1])
def test_number_valid(value):
    assert _number({'probability': value}, 'probability', 0.5, 0, 1) == float(value)


@pytest.mark.parametrize('value', ['bad', None, -0.1, 1.1])
def test_number_invalid(value):
    with pytest.raises(ValueError):
        _number({'probability': value}, 'probability', 0.5, 0, 1)


@pytest.mark.parametrize('terrain,expected', [
    ('Bosque', 'bosque'), ('desert', 'desierto'), ('ciudad', 'urbano'),
    ('dungeon', 'mazmorra')])
def test_terrain_aliases(terrain, expected):
    assert _frontend_config({'terrain_type': terrain})['terrain_type'] == expected


@pytest.mark.parametrize('complexity,expected', [('bajo', 'low'), ('medio', 'medium'), ('alto', 'high')])
def test_complexity_aliases(complexity, expected):
    assert _frontend_config({'complexity': complexity})['complexity'] == expected


@pytest.mark.parametrize('data', [{'terrain_type': 'océano'}, {'complexity': 'extreme'}, {'tile_size': 2}])
def test_frontend_invalid(data):
    with pytest.raises(ValueError):
        _frontend_config(data)


def test_algorithm_defaults_to_cellular_for_dungeon():
    assert _algorithm({}, 'mazmorra') == 'cellular'


def test_algorithm_defaults_to_perlin_for_forest():
    assert _algorithm({}, 'bosque') == 'perlin'


def test_algorithm_explicit_override():
    assert _algorithm({'algorithm': 'cellular'}, 'bosque') == 'cellular'


def test_algorithm_invalid():
    with pytest.raises(ValueError):
        _algorithm({'algorithm': 'invalid'}, 'bosque')
