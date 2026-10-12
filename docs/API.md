# API Rhyzome — endpoints, request y response

Base local: `http://127.0.0.1:5000/api`.

## GET `/health`
No requiere request body. Respuesta `200`: `{"status":"ok","service":"rhyzome-backend"}`.

## POST `/maps/generate`
Genera un mapa procedural. Si no se envía `algorithm`, mazmorra usa `cellular` y los demás terrenos usan `perlin`.

Request Perlin:
```json
{"terrain_type":"bosque","width":50,"height":35,"seed":539989,"algorithm":"perlin","complexity":"medium","tile_size":16,"scale":0.08,"octaves":4,"persistence":0.5,"threshold":0.5}
```

Response `200` (matriz abreviada):
```json
{"status":"success","algorithm":"perlin","seed":539989,"dimensions":{"width":50,"height":35},"parameters":{"scale":0.08,"octaves":4,"persistence":0.5,"threshold":0.5},"ui_config":{"terrain_type":"bosque","complexity":"medium","tile_size":16},"metrics":{"quality":84.2,"accessibility":0.72,"connectivity":0.98},"map":[[0,1],[0,0]]}
```

Request Cellular:
```json
{"terrain_type":"mazmorra","width":50,"height":35,"seed":539989,"algorithm":"cellular","probability":0.45,"iterations":4}
```

## POST `/maps/evaluate`
Request: `{"map":[[0,1],[0,0]]}`. Response `200`: `{"status":"success","metrics":{...}}`. La implementación actual considera `0` celda transitable y `1` obstáculo.

## POST `/maps/generate-adaptive`
Genera varios candidatos y usa el artefacto ML instalado para puntuar/seleccionar un candidato. **Importante:** los `.joblib` incluidos en el ZIP pertenecen al experimento histórico de regresión; no deben presentarse como los clasificadores finales de 10.000 mapas. Para producción deben reemplazarse por artefactos entrenados del experimento final.

## GET `/experiment`
No requiere parámetros. Devuelve dos bloques explícitos:
- `final_experiment`: resultados finales de 10.000 mapas y selección Regresión Logística/MLP.
- `repository_sample`: muestra histórica/auditable incluida físicamente en este ZIP.

## GET `/experiment/maps`
Query: `generator=perlin|cellular`, `limit=1..100`, `offset>=0`.
Ejemplo: `/api/experiment/maps?generator=perlin&limit=20&offset=0`.

## GET `/experiment/maps/{id}`
Ejemplo: `/api/experiment/maps/perlin-0001`. Devuelve el JSON físico de la muestra incluida.

## POST `/auth/register`
Request: `{"email":"usuario@correo.com","password":"secreto"}`. Response `201`: `{"status":"success","user":{"id":"uuid","email":"usuario@correo.com"}}`.

## POST `/auth/login`
Request: `{"email":"usuario@correo.com","password":"secreto"}`. Response `200`: usuario, `access_token` y `refresh_token`. Credenciales inválidas: `401`.

## Errores
`400` parámetros inválidos; `401` autenticación; `404` recurso/artefacto inexistente; `405` método no permitido; `503` modelo ML requerido no disponible.
