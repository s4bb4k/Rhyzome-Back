import numpy as np
import pytest
from application.v2.services.map_service import generate
from application.v2.generators.semantic import generar_A
from application.v2.generators.optimized import generar_B
from application.v2.services.metrics import metricas

@pytest.mark.parametrize('biome',['bosque','desierto','ciudad','mazmorra'])
@pytest.mark.parametrize('variant',['A','B'])
def test_maps_and_repeatability(biome,variant):
    p={'terrain_type':biome,'seed':2026,'width':50,'height':35,'variant':variant}
    a=generate(p);b=generate(p)
    assert a['layers']==b['layers']
    assert a['metrics']==b['metrics']
    assert a['version']=='v2'
    assert a['metrics']['solapamientos_invalidos']==0
    for layer in a['layers'].values():
        assert len(layer)==35 and all(len(row)==50 for row in layer)
    assert set(np.unique(a['layers']['collision'])).issubset({0,1})

@pytest.mark.parametrize('biome',['bosque','desierto','ciudad','mazmorra'])
def test_optimized_connectivity_not_worse(biome):
    a=metricas(generar_A(biome,2026));b=metricas(generar_B(biome,2026)[0])
    assert b['mayor_region_ratio']+1e-10>=a['mayor_region_ratio']

def test_invalid_requests():
    for data in [{'terrain_type':'mar'}, {'width':0}, {'variant':'C'}, {'seed':-1}, {'width':12.5}]:
        with pytest.raises(ValueError):generate(data)

def test_flask_route():
    pytest.importorskip('flask')
    pytest.importorskip('flask_cors')
    from app import create_app
    client=create_app().test_client()
    ok=client.post('/api/v2/maps/generate',json={'terrain_type':'bosque','seed':2026})
    assert ok.status_code==200
    assert ok.json['version']=='v2'
    bad=client.post('/api/v2/maps/generate',json={'terrain_type':'mar'})
    assert bad.status_code==400
    assert client.get('/api/health').status_code==200
