from unittest.mock import Mock
import pytest
from application.services import adaptive_generator_ml as module


def test_load_bundle_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MODEL_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='Ejecuta el experimento'):
        module._load_bundle('cellular')


def test_load_bundle_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MODEL_DIR', tmp_path)
    (tmp_path / 'best_cellular.joblib').touch()
    loader = Mock(return_value={'model_name': 'mock'})
    monkeypatch.setattr(module.joblib, 'load', loader)
    assert module._load_bundle('cellular')['model_name'] == 'mock'
    loader.assert_called_once_with(tmp_path / 'best_cellular.joblib')


@pytest.mark.parametrize('generator,expected', [('cellular', 'generate_cellular'), ('perlin', 'generate_perlin_map')])
def test_candidate_dispatch(monkeypatch, generator, expected):
    cellular = Mock(return_value=[[0]])
    perlin = Mock(return_value=[[1]])
    monkeypatch.setattr(module, 'generate_cellular', cellular)
    monkeypatch.setattr(module, 'generate_perlin_map', perlin)
    config = {'width': 10, 'height': 12, 'probability': 0.4, 'iterations': 3,
              'scale': 0.08, 'octaves': 4, 'persistence': 0.5, 'threshold': 0.5}
    result = module._candidate(generator, config, 42)
    assert result == ([[0]] if generator == 'cellular' else [[1]])
    (cellular if generator == 'cellular' else perlin).assert_called_once()


@pytest.mark.parametrize('generator', ['cellular', 'perlin'])
def test_adaptive_selects_best_prediction(monkeypatch, generator):
    model = Mock()
    model.predict.side_effect = [[0.2], [0.9], [0.4]]
    bundle = {'model': model, 'feature_names': ['accessibility'],
              'model_name': 'mock_model', 'metrics': {'r2': 0.9}}
    monkeypatch.setattr(module, '_load_bundle', Mock(return_value=bundle))
    monkeypatch.setattr(module, '_candidate', Mock(side_effect=[[[0]], [[1]], [[0, 1]]]))
    monkeypatch.setattr(module, 'extract_feature_dict', lambda grid: {'accessibility': 0.5})
    monkeypatch.setattr(module, 'evaluar_mapa', lambda grid: {'quality': 50.0})
    result = module.generate_adaptive_map_ml(generator, {'seed': 42}, attempts=3)
    assert result['predicted_score'] == 0.9
    assert result['map'] == [[1]]
    assert result['algorithm'] == generator
    assert result['selected_model'] == 'mock_model'
    assert result['attempts'] == 3
    assert model.predict.call_count == 3


def test_adaptive_reproducible_seeds(monkeypatch):
    model = Mock()
    model.predict.return_value = [0.5]
    monkeypatch.setattr(module, '_load_bundle', lambda generator: {
        'model': model, 'feature_names': ['accessibility'],
        'model_name': 'mock', 'metrics': {}})
    monkeypatch.setattr(module, 'extract_feature_dict', lambda grid: {'accessibility': 1.0})
    monkeypatch.setattr(module, 'evaluar_mapa', lambda grid: {})
    seeds = []
    def candidate(generator, config, seed):
        seeds.append(seed)
        return [[0]]
    monkeypatch.setattr(module, '_candidate', candidate)
    module.generate_adaptive_map_ml('cellular', {'seed': 12}, attempts=3)
    first = seeds[:]
    seeds.clear()
    module.generate_adaptive_map_ml('cellular', {'seed': 12}, attempts=3)
    assert seeds == first


@pytest.mark.xfail(strict=True, reason='La función no valida attempts=0 y falla al expandir best=None')
def test_adaptive_rejects_zero_attempts(monkeypatch):
    monkeypatch.setattr(module, '_load_bundle', lambda generator: {})
    with pytest.raises(ValueError, match='attempts'):
        module.generate_adaptive_map_ml('cellular', {'seed': 42}, attempts=0)
