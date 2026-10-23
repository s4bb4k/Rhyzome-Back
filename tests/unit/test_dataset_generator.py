import csv
import json
import random
from unittest.mock import Mock
import pytest
from application.services import dataset_generator as module


@pytest.mark.parametrize('generator', ['cellular', 'perlin'])
def test_configuration_ranges(generator):
    width, height, seed, params = module._configuration(generator, random.Random(42))
    assert 30 <= width <= 50
    assert 30 <= height <= 50
    assert 1 <= seed <= 2_147_483_647
    if generator == 'cellular':
        assert 0.38 <= params['probability'] <= 0.58
        assert 2 <= params['iterations'] <= 7
    else:
        assert 0.035 <= params['scale'] <= 0.12
        assert 2 <= params['octaves'] <= 6
        assert 0.35 <= params['persistence'] <= 0.7
        assert 0.42 <= params['threshold'] <= 0.58


@pytest.mark.parametrize('generator,expected', [('cellular', 'cellular'), ('perlin', 'perlin')])
def test_generate_dispatch(monkeypatch, generator, expected):
    cell = Mock(return_value=[[0]])
    perlin = Mock(return_value=[[1]])
    monkeypatch.setattr(module, 'generate_cellular', cell)
    monkeypatch.setattr(module, 'generate_perlin_map', perlin)
    params = {'probability': 0.45, 'iterations': 4, 'scale': 0.08,
              'octaves': 4, 'persistence': 0.5, 'threshold': 0.5}
    result = module._generate(generator, 10, 10, 42, params)
    assert result == ([[0]] if generator == 'cellular' else [[1]])
    (cell if generator == 'cellular' else perlin).assert_called_once()


def test_dataset_writes_json_csv_metadata_to_temporary_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_generate', lambda *args: [[0, 1], [0, 0]])
    rows = module.generate_dataset(tmp_path, samples_per_generator=2, experiment_seed=42)
    assert len(rows) == 4
    assert {row['generator'] for row in rows} == {'cellular', 'perlin'}
    with (tmp_path / 'dataset_index.csv').open(encoding='utf-8', newline='') as handle:
        index = list(csv.DictReader(handle))
    assert len(index) == 4
    metadata = json.loads((tmp_path / 'dataset_metadata.json').read_text(encoding='utf-8'))
    assert metadata['total_maps'] == 4
    assert metadata['experiment_seed'] == 42
    for row in rows:
        record = json.loads((tmp_path / row['json_path']).read_text(encoding='utf-8'))
        assert record['id'] == row['id']
        assert record['map'] == [[0, 1], [0, 0]]
        assert record['target_score'] == row['target_score']


def test_dataset_same_seed_same_records(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_generate', lambda *args: [[0, 0], [1, 0]])
    first = module.generate_dataset(tmp_path / 'first', 1, 77)
    second = module.generate_dataset(tmp_path / 'second', 1, 77)
    assert first == second


def test_dataset_zero_samples_creates_empty_index(tmp_path):
    rows = module.generate_dataset(tmp_path, samples_per_generator=0)
    assert rows == []
    metadata = json.loads((tmp_path / 'dataset_metadata.json').read_text(encoding='utf-8'))
    assert metadata['total_maps'] == 0
