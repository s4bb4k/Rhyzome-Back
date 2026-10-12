# Guía de sustentación — Experimentación Rhyzome

## 1. Qué se experimentó

Rhyzome separa dos problemas. **Perlin Noise y Autómatas Celulares generan mapas**; no son los modelos de Machine Learning. Después de generar los mapas se extraen características estructurales y el experimento de ML aprende a reconocer si la complejidad corresponde a **Bajo, Medio o Alto**.

El experimento final trabajó con **10.000 mapas**: 7.500 de Perlin Noise para Bosque, Desierto y Ciudad, y 2.500 de Autómatas Celulares para Mazmorra.

## 2. Flujo de la experimentación

1. Se varían semillas y parámetros del generador para producir mapas distintos.
2. Cada mapa se representa como una matriz binaria y se calculan métricas estructurales.
3. Se obtiene un `complexity_score` normalizando métricas con percentiles P5/P95 y combinándolas en una escala 0–100.
4. El score se divide en terciles para producir las clases Bajo, Medio y Alto.
5. Los modelos ML se entrenan/evalúan para predecir esas clases.
6. Se usa validación cruzada estratificada de 5 folds y semillas 42, 123 y 2026 para evitar depender de una sola partición.
7. El criterio principal de selección es **F1 Macro**, complementado por Accuracy, ROC-AUC, estabilidad entre semillas y tiempo de inferencia.
8. El resultado final selecciona **Regresión Logística para Perlin** y **MLP para Autómatas Celulares**.

## 3. Qué significa cada familia de métricas

Las métricas del mapa responden si la estructura es abierta/cerrada, conectada/fragmentada y simple/compleja. `floor_ratio` mide accesibilidad; `largest_region_ratio`, conectividad; `n_regions`, fragmentación; `edge_density` y `local_variance`, cambios estructurales; las entropías describen distribución espacial; `perimeter_area_ratio`, irregularidad; y `complexity_score` resume seis señales normalizadas.

Las métricas de ML responden otra pregunta: qué tan bien el modelo reconoce Bajo/Medio/Alto. F1 Macro es principal porque da importancia equilibrada a las tres clases.

## 4. Resultados que debes memorizar

| Generador | Modelo seleccionado | CV F1 Macro | Accuracy | F1 prueba | ROC-AUC | Desv. semillas |
|---|---|---:|---:|---:|---:|---:|
| Perlin Noise | Regresión Logística | 0.8203 | 0.8218 | 0.8207 | 0.9439 | 0.0008 |
| Autómatas Celulares | MLP | 0.8054 | 0.8293 | 0.8304 | 0.9467 | 0.0195 |

## 5. Cómo se integró al backend

El archivo `experiment/final_experiment_results.json` es el manifiesto de resultados finales. `GET /api/experiment` lo expone junto con la muestra auditable que sí está físicamente incluida en el repositorio. Esta separación evita confundir la evidencia final de 10.000 mapas con el dataset histórico de 1.000 JSON que venía empaquetado en esta versión del proyecto.

La recomendación funcional queda: Bosque/Desierto/Ciudad → Perlin + Regresión Logística; Mazmorra → Autómatas Celulares + MLP.

## 6. Respuesta corta para sustentación

> “Primero generamos mapas con Perlin Noise y Autómatas Celulares. Sobre cada mapa extraemos características objetivas de accesibilidad, conectividad, fragmentación y complejidad. Con esas características evaluamos modelos de Machine Learning para clasificar el nivel Bajo, Medio o Alto. No elegimos el modelo por una sola partición: usamos validación cruzada estratificada de cinco folds y repetimos con tres semillas. El criterio principal fue F1 Macro. En 10.000 mapas, Regresión Logística fue la mejor opción para Perlin y MLP para Autómatas Celulares. Esos resultados quedaron incorporados al backend como evidencia y como regla de selección por generador.”

## 7. Preguntas probables

**¿Por qué no usar un solo modelo para todo?** Porque Perlin y Autómatas producen estructuras diferentes y la evaluación independiente mostró ganadores distintos.

**¿Por qué F1 Macro?** Porque Bajo, Medio y Alto importan por igual; evita que una clase dominante o más fácil oculte errores en las otras.

**¿Por qué validación cruzada?** Para reducir el riesgo de escoger un modelo favorecido accidentalmente por una única división entrenamiento/prueba.

**¿Un mapa con 100 % de accesibilidad es necesariamente mejor?** No. Puede ser demasiado abierto y simple. La calidad estructural se interpreta combinando accesibilidad, conectividad, fragmentación y complejidad.

**¿Los 10.000 JSON están en este ZIP?** No. Este paquete contiene una muestra auditable histórica de 1.000 mapas. El resultado final de 10.000 está registrado como manifiesto experimental y documentado; para afirmar que los 10.000 archivos están versionados, deben incorporarse físicamente al repositorio o almacenarse como artefacto externo.
