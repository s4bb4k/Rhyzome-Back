import csv
import json
from pathlib import Path

from flask import Blueprint, jsonify, request

experiment_bp = Blueprint("experiment", __name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiment"


def _read_json(path):
    if not path.exists():
        raise FileNotFoundError("Primero debe ejecutarse el experimento")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@experiment_bp.get("")
def summary():
    try:
        metadata = _read_json(EXPERIMENT_DIR / "dataset" / "dataset_metadata.json")
        results = _read_json(EXPERIMENT_DIR / "artifacts" / "experiment_results.json")
        return jsonify({"status": "success", "dataset": metadata, "experiment": results})
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404


@experiment_bp.get("/maps")
def list_maps():
    generator = request.args.get("generator")
    if generator not in (None, "cellular", "perlin"):
        return jsonify({"status": "error", "message": "Generador no válido"}), 400
    try:
        limit = min(max(int(request.args.get("limit", 20)), 1), 100)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"status": "error", "message": "limit y offset deben ser enteros"}), 400

    index_path = EXPERIMENT_DIR / "dataset" / "dataset_index.csv"
    if not index_path.exists():
        return jsonify({"status": "error", "message": "Dataset no disponible"}), 404
    with index_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if generator:
        rows = [row for row in rows if row["generator"] == generator]
    return jsonify({"status": "success", "total": len(rows), "offset": offset,
                    "limit": limit, "maps": rows[offset:offset + limit]})


@experiment_bp.get("/maps/<map_id>")
def get_map(map_id):
    if not (map_id.startswith("cellular-") or map_id.startswith("perlin-")):
        return jsonify({"status": "error", "message": "Identificador no válido"}), 400
    generator = map_id.split("-", 1)[0]
    if not map_id.replace("-", "").isalnum():
        return jsonify({"status": "error", "message": "Identificador no válido"}), 400
    path = EXPERIMENT_DIR / "dataset" / "maps" / generator / f"{map_id}.json"
    try:
        return jsonify({"status": "success", "training_map": _read_json(path)})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "Mapa no encontrado"}), 404
