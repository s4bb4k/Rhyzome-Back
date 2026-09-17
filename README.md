# Rhizome API

API REST en Flask para generar y evaluar mapas procedurales 2D, seleccionar el mejor candidato mediante ML y consultar la evidencia del experimento. React + Vite se ejecuta por separado.

## Inicio rápido

Requiere Python 3.12. En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Base URL: `http://127.0.0.1:5000/api`. Todas las respuestas son JSON. Los éxitos contienen `status: "success"` (salvo health: `"ok"`) y los errores: `{"status":"error","message":"..."}`.

## Endpoints

| Método | Ruta | Resultado |
| --- | --- | --- |
| GET | `/health` | Estado del backend |
| POST | `/maps/generate` | Mapa procedural |
| POST | `/maps/evaluate` | Métricas de una matriz |
| POST | `/maps/generate-adaptive` | Mejor candidato según ML |
| GET | `/experiment` | Dataset, resultados y ganadores |
| GET | `/experiment/maps` | Mapas de entrenamiento paginados |
| GET | `/experiment/maps/{id}` | Mapa de entrenamiento completo |
| POST | `/auth/login` | Inicio de sesión Supabase |
| POST | `/auth/register` | Registro Supabase |

## Contrato del dashboard React

La pantalla puede enviar nombres en español o inglés. `terrain_type` también puede llamarse `map_type`. Si omite `algorithm`, se usa `cellular` para mazmorra y `perlin` para los otros terrenos.

| Control | Campo JSON | Tipo / valores | Requerido | Defecto |
| --- | --- | --- | --- | --- |
| Tipo de mapa | `terrain_type` | bosque, desierto, urbano/ciudad, mazmorra (o equivalentes en inglés) | No | bosque |
| Ancho | `width` | entero 10–200 | No | 50 |
| Alto | `height` | entero 10–200 | No | 35 |
| Semilla | `seed` | entero 0–2147483647 o null | No | aleatoria |
| Algoritmo | `algorithm` | perlin o cellular | No | recomendado |
| Complejidad | `complexity` | low/medium/high o bajo/medio/alto | No | medium |
| Tile | `tile_size` | entero 4–64 | No | 16 |

`complexity` y `tile_size` son configuración visual y vuelven normalizados en `ui_config`; no cambian las fórmulas del generador. Calidad y accesibilidad son resultados de `metrics`, no entradas.

## POST `/maps/generate`

Parámetros Perlin:

| Campo | Tipo | Rango | Defecto |
| --- | --- | --- | --- |
| `scale` | decimal | 0.005–1.0 | 0.08 |
| `octaves` | entero | 1–8 | 4 |
| `persistence` | decimal | 0.1–1.0 | 0.5 |
| `threshold` | decimal | 0.0–1.0 | 0.5 |

Parámetros celulares:

| Campo | Tipo | Rango | Defecto |
| --- | --- | --- | --- |
| `probability` | decimal | 0.0–1.0 | 0.45 |
| `iterations` | entero | 0–20 | 4 |

Entrada compatible con la pantalla:

```json
{
  "terrain_type": "bosque",
  "width": 50,
  "height": 35,
  "seed": 539989,
  "algorithm": "perlin",
  "complexity": "medium",
  "tile_size": 16,
  "scale": 0.08,
  "octaves": 4,
  "persistence": 0.5,
  "threshold": 0.5
}
```

Respuesta `200` (matriz abreviada):

```json
{
  "status": "success",
  "algorithm": "perlin",
  "seed": 539989,
  "dimensions": {"width": 50, "height": 35},
  "parameters": {"scale": 0.08, "octaves": 4, "persistence": 0.5, "threshold": 0.5},
  "ui_config": {"terrain_type": "bosque", "complexity": "medium", "tile_size": 16},
  "metrics": {
    "quality": 84.2,
    "accessibility": 0.72,
    "connectivity": 0.98,
    "obstacle_density": 0.28,
    "connected_components": 2,
    "largest_open_area": 0.70,
    "border_obstacle_ratio": 0.66,
    "balance": 0.94
  },
  "map": [[0, 0, 1], [0, 1, 1]]
}
```

La matriz real tiene `height` filas y `width` columnas; `0` es transitable y `1` obstáculo. Los mismos parámetros y semilla producen el mismo mapa.

## POST `/maps/evaluate`

Recibe una matriz rectangular, no vacía y formada solo por 0 y 1:

```json
{"map": [[0, 1], [0, 0]]}
```

Respuesta `200`: `{"status":"success","metrics":{...}}`, con las ocho métricas anteriores.

## POST `/maps/generate-adaptive`

Acepta los parámetros de generación y `attempts` (entero 1–50, defecto 10). Devuelve el candidato con mayor puntuación predicha. La respuesta incluye `map`, `seed`, `metrics`, `predicted_score`, `algorithm`, `selected_model`, `model_metrics`, `attempts`, `parameters` y `ui_config`. Retorna `503` si falta el modelo ML.

Ejemplo celular:

```json
{
  "terrain_type": "mazmorra",
  "algorithm": "cellular",
  "width": 50,
  "height": 35,
  "seed": 539989,
  "complexity": "high",
  "tile_size": 16,
  "probability": 0.45,
  "iterations": 4,
  "attempts": 10
}
```

## API del experimento

`GET /experiment` no recibe parámetros y devuelve `dataset` y `experiment`. Se incluyen 500 mapas celulares y 500 Perlin, y resultados de Random Forest, SVM y MLP.

`GET /experiment/maps` acepta:

| Query param | Valores | Defecto |
| --- | --- | --- |
| `generator` | cellular, perlin u omitido | ambos |
| `limit` | entero 1–100 | 20 |
| `offset` | entero >= 0 | 0 |

Ejemplo: `/api/experiment/maps?generator=cellular&limit=20&offset=0`. Responde con `status`, `total`, `offset`, `limit` y `maps`.

`GET /experiment/maps/{id}` usa identificadores como `cellular-0001` o `perlin-0001`. Responde con `training_map`, que incluye matriz, parámetros, variables extraídas y puntuación objetivo.

## Autenticación

`POST /auth/register` recibe `email` y `password` obligatorios. Devuelve `201` y `{"status":"success","user":{"id":"uuid","email":"..."}}`.

`POST /auth/login` recibe los mismos campos. Devuelve `user`, `access_token` y `refresh_token`; credenciales inválidas dan `401`. Requiere `SUPABASE_URL` y `SUPABASE_KEY` en `.env`.

## Integración React + Vite

`.env` del frontend:

```dotenv
VITE_API_URL=http://127.0.0.1:5000/api
```

```javascript
const response = await fetch(`${import.meta.env.VITE_API_URL}/maps/generate`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(formValues),
});
const data = await response.json();
if (!response.ok) throw new Error(data.message ?? "No se pudo generar el mapa");
setMap(data.map);
setMetrics(data.metrics);
```

El backend permite `http://localhost:5173`. Para Vercel configure:

```dotenv
CORS_ORIGINS=http://localhost:5173,https://mi-rhizome.vercel.app
```

## Errores y pruebas

| Código | Significado |
| --- | --- |
| 400 | Tipo, rango o valor de parámetro inválido |
| 401 | Credenciales inválidas |
| 404 | Ruta, dataset o mapa inexistente |
| 405 | Método HTTP no permitido |
| 503 | Modelo ML no disponible |

```powershell
python -m unittest discover -s tests -v
python -m application.services.train_model --samples-per-generator 500
```

Las pruebas cubren salud, ambos generadores, reproducibilidad, dimensiones, contrato del dashboard, recomendación para mazmorra, evaluación, validaciones, evidencia experimental y CORS de Vite.
