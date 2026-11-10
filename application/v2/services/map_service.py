"""Contrato de generación V2. El ML V1 no se invoca automáticamente."""
import secrets
import numpy as np
from application.v2.generators.semantic import generar_A, TERRAIN, OBJECT, BIOMAS
from application.v2.generators.optimized import generar_B
from application.v2.services.metrics import metricas

ALIASES = {'forest':'bosque','desert':'desierto','urban':'ciudad','urbano':'ciudad','dungeon':'mazmorra'}

def _int(data, key, default, minimum, maximum):
    raw=data.get(key,default)
    if isinstance(raw,bool):raise ValueError(f"'{key}' debe ser entero")
    try:
        v=int(raw)
        if str(raw).strip()!=str(v):raise ValueError()
    except (TypeError,ValueError):raise ValueError(f"'{key}' debe ser entero")
    if not minimum<=v<=maximum:raise ValueError(f"'{key}' debe estar entre {minimum} y {maximum}")
    return v

def generate(data):
    if not isinstance(data,dict):raise ValueError('El cuerpo debe ser un objeto JSON')
    biome=str(data.get('terrain_type',data.get('biome','bosque'))).lower().strip()
    biome=ALIASES.get(biome,biome)
    if biome not in BIOMAS:raise ValueError("'terrain_type' debe ser bosque, desierto, ciudad o mazmorra")
    width=_int(data,'width',50,10,200);height=_int(data,'height',35,10,200)
    seed=_int(data,'seed',secrets.randbelow(2147483648),0,2147483647)
    version=str(data.get('variant','B')).upper()
    if version not in ('A','B'):raise ValueError("'variant' debe ser A o B")
    if version=='A':m=generar_A(biome,seed,height,width);cleared=0
    else:m,cleared=generar_B(biome,seed,height,width)
    payload={'status':'success','version':'v2','schema_version':'2.1-experimental',
        'variant':version,'terrain_type':biome,'biome':biome,'seed':seed,
        'width':width,'height':height,'dimensions':{'width':width,'height':height},
        'terrain_catalog':TERRAIN,'object_catalog':OBJECT,
        'layers':{key:m[key].astype(int).tolist() for key in ('terrain','objects','collision')},
        'metrics':metricas(m),'repaired_cells':cleared,
        'generator_note':'ruido_suave experimental; no equivale al Perlin de V1'}
    return payload
