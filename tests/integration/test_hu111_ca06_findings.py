
import pytest

from application.generators.perlin_noise import generate_perlin_map
from application.services.feature_extractor import extract_feature_dict
from application.services.evaluator import evaluate


# ==========================================================
# HU-111 / CA-06 / H-01
# Verificar el comportamiento del parámetro scale en Perlin
# ==========================================================

def test_h01_scale_no_modifica_mapa_perlin():
    config = {
        "width": 50,
        "height": 35,
        "octaves": 4,
        "persistence": 0.5,
        "threshold": 0.5,
        "seed": 12345,
    }

    mapa_escala_baja = generate_perlin_map(
        **config,
        scale=0.02,
    )

    mapa_escala_alta = generate_perlin_map(
        **config,
        scale=0.20,
    )

    assert mapa_escala_baja == mapa_escala_alta, (
        "El comportamiento de scale ha cambiado: "
        "las matrices ahora son diferentes."
    )

    print("\n[H-01] RESULTADO")
    print("scale=0.02 y scale=0.20 producen la misma matriz.")
    print("Hallazgo confirmado para esta configuracion.")


# ==========================================================
# HU-111 / CA-06 / H-02
# Verificar el cálculo de border_obstacle_ratio
# ==========================================================

def test_h02_border_obstacle_ratio_constante():
    # Todos los bordes están bloqueados.
    mapa_bordes_bloqueados = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    # Todos los bordes son transitables.
    mapa_bordes_abiertos = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    features_bloqueados = extract_feature_dict(
        mapa_bordes_bloqueados
    )
    features_abiertos = extract_feature_dict(
        mapa_bordes_abiertos
    )

    ratio_bloqueados = features_bloqueados[
        "border_obstacle_ratio"
    ]
    ratio_abiertos = features_abiertos[
        "border_obstacle_ratio"
    ]

    assert ratio_bloqueados == 0.0
    assert ratio_abiertos == 0.0

    print("\n[H-02] RESULTADO")
    print("Bordes bloqueados:", ratio_bloqueados)
    print("Bordes abiertos:", ratio_abiertos)
    print("La métrica devuelve cero en ambos escenarios.")


# ==========================================================
# HU-111 / CA-06 / H-02
# Verificar el peso de la métrica en la evaluación
# ==========================================================

def test_h02_border_obstacle_ratio_influye_en_calidad():
    mapa = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    features = extract_feature_dict(mapa)

    calidad_actual = evaluate(features)

    # Simular exclusivamente un cambio en esta métrica.
    features_alternativas = {
        **features,
        "border_obstacle_ratio": 1.0,
    }

    calidad_alternativa = evaluate(
        features_alternativas
    )

    diferencia = calidad_alternativa - calidad_actual

    assert diferencia == pytest.approx(10.0)

    print("\n[H-02] IMPACTO EN CALIDAD")
    print("Calidad con ratio 0:", calidad_actual)
    print("Calidad con ratio 1:", calidad_alternativa)
    print("Diferencia:", round(diferencia, 4), "puntos")


