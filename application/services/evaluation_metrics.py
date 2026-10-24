from application.services.evaluator import evaluate
from application.services.feature_extractor import extract_feature_dict


def evaluar_mapa(mapa, ml_features=None):
    """
    Evalúa un mapa en términos de:
    - calidad
    - coherencia
    - eficiencia

    HU-78:
    Reutiliza características ML cuando
    están disponibles.
    """

    features = extract_feature_dict(
        mapa,
        ml_features=ml_features
    )

    return {
        "quality": evaluate(features),
        "accessibility": round(
            features["accessibility"], 4
        ),
        "connectivity": round(
            features["connectivity"], 4
        ),
        "obstacle_density": round(
            features["obstacle_density"], 4
        ),
        "connected_components": (
            features["connected_components"]
        ),
        "largest_open_area": round(
            features["largest_open_area"], 4
        ),
        "border_obstacle_ratio": round(
            features["border_obstacle_ratio"], 4
        ),
        "balance": round(
            features["balance"], 4
        ),
    }