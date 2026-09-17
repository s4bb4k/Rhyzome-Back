# Evidencia del experimento ML

Este directorio permite auditar y reproducir el entrenamiento de Rhizome.

- `dataset/dataset_metadata.json`: configuración y cantidad total de mapas.
- `dataset/dataset_index.csv`: índice de los 1.000 mapas y sus características.
- `dataset/maps/cellular/`: 500 mapas de autómatas celulares en JSON.
- `dataset/maps/perlin/`: 500 mapas Perlin en JSON.
- `artifacts/split_*.json`: identificadores usados para entrenamiento y prueba.
- `artifacts/model_comparison.csv`: métricas de Random Forest, SVM y MLP.
- `artifacts/experiment_results.json`: resultados y ganador por generador.
- `artifacts/models/`: modelos entrenados y copia del ganador de cada generador.

Cada JSON contiene identificador, semilla, dimensiones, parámetros, siete
características, puntuación objetivo y la matriz completa del mapa.

## Reproducir el experimento

```bash
python -m application.services.train_model --samples-per-generator 500
```

La semilla del experimento es `20260909`, la división entrenamiento/prueba es
80/20 y utiliza `random_state=42`. El ganador se elige por mayor R² y, en caso
de empate, por menor MAE.

SVM y MLP se entrenan dentro de un `Pipeline` con `StandardScaler`; así la
normalización aprendida con el conjunto de entrenamiento queda guardada junto
al modelo y se reutiliza durante la predicción.
