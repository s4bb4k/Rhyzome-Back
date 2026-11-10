# Rhyzome V2 — Integración de experimentos 2 y 3

## Qué contiene

Este paquete incorpora la **lógica A** de generación semántica del Experimento 2 y la **lógica B** de optimización del Experimento 3. Los archivos originales V1 se conservan, salvo el registro del nuevo Blueprint en `app.py`.

- `application/v2/generators/semantic.py`: biomas y capas del Experimento 2 (línea base A).
- `application/v2/services/connectivity.py`: reparación de conectividad del Experimento 3.
- `application/v2/generators/optimized.py`: reglas B de los cuatro biomas.
- `application/v2/services/metrics.py`: métricas propias de V2.
- `application/v2/services/map_service.py`: validación y respuesta JSON.
- `application/v2/routes/map_routes.py`: endpoint Flask aislado.
- `tests/v2/`: pruebas de generación, consistencia y API.

## Cómo ejecutarlo

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python app.py
```

Petición de ejemplo:

```bash
curl -X POST http://127.0.0.1:5000/api/v2/maps/generate \
  -H 'Content-Type: application/json' \
  -d '{"terrain_type":"bosque","seed":2026,"width":50,"height":35,"variant":"B"}'
```

`variant=A` reproduce las reglas semánticas originales; `variant=B` aplica optimización. Respuesta: `layers.terrain`, `layers.objects`, `layers.collision`, catálogos y métricas. Para pruebas: `python -m pytest tests/v2 -q`.

## Limitaciones que deben conservarse en el informe

1. El ruido de los experimentos es **ruido suavizado interpolado**, no el Perlin real de V1; la migración al generador Perlin existente requiere otra validación.
2. Los modelos ML de V1 **no se reemplazan ni se conectan automáticamente** a V2. El Experimento 1 sirve como evidencia de selección preliminar; falta evaluar/reentrenar con mapas V2.
3. No hay sprites, PixiJS ni renderizado pixel art en este backend.
4. La reparación de conectividad puede crear caminos poco naturales: requiere evaluación visual y pruebas con más tamaños y semillas.
5. Las métricas no constituyen validación completa de jugabilidad; las reglas se extrajeron del notebook experimental y todavía requieren endurecimiento para producción.
6. No se añaden autenticación ni persistencia específicas de V2; deben integrarse según los requisitos de la aplicación.

## Compatibilidad

La API V1 `/api/maps/*` sigue registrada con el mismo Blueprint. No se modificaron generadores ni servicios V1. Se excluyeron del ZIP de entrega `.env`, `.git`, `.venv`, cachés e informes generados para evitar compartir secretos y archivos locales. Los artefactos de modelos existentes se mantienen en `experiment/artifacts/models`.
