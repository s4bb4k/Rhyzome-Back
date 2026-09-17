from flask import Blueprint, jsonify, request

from application.generators.cellular_automata import generate_map as generate_cellular_map
from application.generators.perlin_noise import generate_perlin_map
from application.services.evaluation_metrics import evaluar_mapa

map_bp = Blueprint("maps", __name__)

TERRAIN_ALIASES = {
    "forest": "bosque", "bosque": "bosque",
    "desert": "desierto", "desierto": "desierto",
    "urban": "urbano", "urbano": "urbano", "ciudad": "urbano",
    "dungeon": "mazmorra", "mazmorra": "mazmorra",
}


def _integer(data, field, default, minimum, maximum):
    try:
        value = int(data.get(field, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field}' debe ser un número entero") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"'{field}' debe estar entre {minimum} y {maximum}")
    return value


def _number(data, field, default, minimum, maximum):
    try:
        value = float(data.get(field, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field}' debe ser un número") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"'{field}' debe estar entre {minimum} y {maximum}")
    return value


def _frontend_config(data):
    """Valida campos visuales enviados por el dashboard de React."""
    raw_terrain = str(data.get("terrain_type", data.get("map_type", "bosque"))).lower()
    terrain_type = TERRAIN_ALIASES.get(raw_terrain)
    if terrain_type is None:
        raise ValueError("'terrain_type' debe ser bosque, desierto, urbano o mazmorra")
    complexity = str(data.get("complexity", "medium")).lower()
    complexity_aliases = {
        "low": "low", "bajo": "low", "medium": "medium", "medio": "medium",
        "high": "high", "alto": "high",
    }
    if complexity not in complexity_aliases:
        raise ValueError("'complexity' debe ser low, medium o high")
    return {
        "terrain_type": terrain_type,
        "complexity": complexity_aliases[complexity],
        "tile_size": _integer(data, "tile_size", 16, 4, 64),
    }


def _algorithm(data, terrain_type):
    raw = data.get("algorithm")
    algorithm = ("cellular" if terrain_type == "mazmorra" else "perlin") if raw in (None, "") else str(raw).lower()
    if algorithm not in ("cellular", "perlin"):
        raise ValueError("'algorithm' debe ser 'cellular' o 'perlin'")
    return algorithm


@map_bp.post("/generate")
def generate_map():
    """Genera un mapa mediante autómatas celulares o ruido Perlin."""
    try:
        data = request.get_json(silent=True) or {}
        frontend = _frontend_config(data)
        algorithm = _algorithm(data, frontend["terrain_type"])
        width = _integer(data, "width", 50, 10, 200)
        height = _integer(data, "height", 35, 10, 200)
        seed = data.get("seed")
        seed = None if seed in (None, "") else _integer(data, "seed", 0, 0, 2_147_483_647)

        if algorithm == "cellular":
            probability = _number(data, "probability", 0.45, 0.0, 1.0)
            iterations = _integer(data, "iterations", 4, 0, 20)
            grid = generate_cellular_map(width, height, probability, iterations, seed)
            parameters = {"probability": probability, "iterations": iterations}
        elif algorithm == "perlin":
            scale = _number(data, "scale", 0.08, 0.005, 1.0)
            octaves = _integer(data, "octaves", 4, 1, 8)
            persistence = _number(data, "persistence", 0.5, 0.1, 1.0)
            threshold = _number(data, "threshold", 0.5, 0.0, 1.0)
            grid = generate_perlin_map(
                width, height, scale, octaves, persistence, seed, threshold
            )
            parameters = {
                "scale": scale,
                "octaves": octaves,
                "persistence": persistence,
                "threshold": threshold,
            }
        return jsonify({
            "status": "success",
            "algorithm": algorithm,
            "seed": seed,
            "dimensions": {"width": width, "height": height},
            "parameters": parameters,
            "ui_config": frontend,
            "metrics": evaluar_mapa(grid),
            "map": grid,
        })
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@map_bp.post("/evaluate")
def evaluate_map():
    try:
        data = request.get_json(silent=True) or {}
        grid = data.get("map")
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            raise ValueError("'map' debe ser una matriz no vacía")
        width = len(grid[0])
        if width == 0 or any(not isinstance(row, list) or len(row) != width for row in grid):
            raise ValueError("Todas las filas de 'map' deben tener el mismo tamaño")
        if any(cell not in (0, 1) for row in grid for cell in row):
            raise ValueError("El mapa solo puede contener valores 0 y 1")
        return jsonify({"status": "success", "metrics": evaluar_mapa(grid)})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@map_bp.post("/generate-adaptive")
def generate_adaptive():
    try:
        from application.services.adaptive_generator_ml import generate_adaptive_map_ml

        data = request.get_json(silent=True) or {}
        frontend = _frontend_config(data)
        algorithm = _algorithm(data, frontend["terrain_type"])
        seed = data.get("seed")
        seed = None if seed in (None, "") else _integer(data, "seed", 0, 0, 2_147_483_647)
        config = {
            "width": _integer(data, "width", 50, 10, 200),
            "height": _integer(data, "height", 35, 10, 200),
            "seed": seed,
        }
        if algorithm == "cellular":
            config.update({
                "probability": _number(data, "probability", 0.45, 0.0, 1.0),
                "iterations": _integer(data, "iterations", 4, 0, 20),
            })
        else:
            config.update({
                "scale": _number(data, "scale", 0.08, 0.005, 1.0),
                "octaves": _integer(data, "octaves", 4, 1, 8),
                "persistence": _number(data, "persistence", 0.5, 0.1, 1.0),
                "threshold": _number(data, "threshold", 0.5, 0.0, 1.0),
            })
        attempts = _integer(data, "attempts", 10, 1, 50)
        result = generate_adaptive_map_ml(algorithm, config, attempts)
        return jsonify({"status": "success", "ui_config": frontend, **result})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except (FileNotFoundError, ImportError) as exc:
        return jsonify({"status": "error", "message": f"Modelo ML no disponible: {exc}"}), 503
