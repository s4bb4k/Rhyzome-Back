from flask import Blueprint, jsonify, request

auth_bp = Blueprint("auth", __name__)


def _credentials():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", ""))
    if not email or not password:
        raise ValueError("Los campos 'email' y 'password' son obligatorios")
    return email, password


@auth_bp.post("/login")
def login():
    try:
        from services.auth_service import login_user

        email, password = _credentials()
        response = login_user(email, password)
        session = getattr(response, "session", None)
        return jsonify({
            "status": "success",
            "user": {"id": response.user.id, "email": response.user.email},
            "access_token": session.access_token if session else None,
            "refresh_token": session.refresh_token if session else None,
        })
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 401


@auth_bp.post("/register")
def register():
    try:
        from services.auth_service import register_user

        email, password = _credentials()
        response = register_user(email, password)
        return jsonify({
            "status": "success",
            "user": {"id": response.user.id, "email": response.user.email},
        }), 201
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
