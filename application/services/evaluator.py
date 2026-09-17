def evaluate(features):
    """Calcula una puntuación objetivo de jugabilidad entre 0 y 100."""
    if not isinstance(features, dict):
        names = ["connectivity", "accessibility", "obstacle_density",
                 "connected_components", "largest_open_area",
                 "border_obstacle_ratio", "balance"]
        features = dict(zip(names, features))

    fragmentation = min(features["connected_components"] / 15.0, 1.0)
    score = 100 * (
        0.30 * features["connectivity"]
        + 0.20 * features["balance"]
        + 0.20 * features["largest_open_area"]
        + 0.15 * (1.0 - abs(features["obstacle_density"] - 0.35))
        + 0.10 * features["border_obstacle_ratio"]
        + 0.05 * (1.0 - fragmentation)
    )
    return round(max(0.0, min(100.0, score)), 4)
