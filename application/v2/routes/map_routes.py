from flask import Blueprint, jsonify, request

from application.v2.services.map_service import generate
from application.v2.services.adaptive_service import generate_adaptive

v2_map_bp = Blueprint('v2_maps', __name__)


@v2_map_bp.post('/generate')
def generate_v2():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'status': 'error', 'message': 'Se requiere JSON válido'}), 400
    try:
        return jsonify(generate(data))
    except ValueError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400

@v2_map_bp.post("/generate-adaptive")
def generate_adaptive_v2():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "status": "error",
            "message": "Se requiere JSON válido",
        }), 400

    try:
        result = generate_adaptive(data)
        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 400