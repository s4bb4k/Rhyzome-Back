import os

from flask import Flask, jsonify
from flask_cors import CORS

from application.routes.auth_routes import auth_bp
from application.routes.experiment_routes import experiment_bp
from application.routes.map_routes import map_bp


def create_app():
    """Crea la API de Rhizome sin servir vistas HTML."""
    app = Flask(__name__)
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    CORS(app, resources={r"/api/*": {"origins": origins}})
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(map_bp, url_prefix="/api/maps")
    app.register_blueprint(experiment_bp, url_prefix="/api/experiment")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "rhizome-api"})

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"status": "error", "message": "Recurso no encontrado"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"status": "error", "message": "Método HTTP no permitido"}), 405

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
