
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_auth_service(monkeypatch):
    """Simula Supabase sin realizar llamadas reales."""
    service = types.ModuleType("services.auth_service")
    service.login_user = Mock()
    service.register_user = Mock()

    monkeypatch.setitem(
        sys.modules,
        "services.auth_service",
        service,
    )

    return service


def test_login_success(client, mock_auth_service):
    mock_auth_service.login_user.return_value = SimpleNamespace(
        user=SimpleNamespace(
            id="user-123",
            email="test@example.com",
        ),
        session=SimpleNamespace(
            access_token="fake-access-token",
            refresh_token="fake-refresh-token",
        ),
    )

    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "Password123",
        },
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["user"]["id"] == "user-123"
    assert data["access_token"] == "fake-access-token"
    assert data["refresh_token"] == "fake-refresh-token"

    mock_auth_service.login_user.assert_called_once_with(
        "test@example.com",
        "Password123",
    )


def test_login_without_session(client, mock_auth_service):
    mock_auth_service.login_user.return_value = SimpleNamespace(
        user=SimpleNamespace(
            id="user-123",
            email="test@example.com",
        ),
        session=None,
    )

    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "Password123",
        },
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["access_token"] is None
    assert data["refresh_token"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"email": "", "password": "Password123"},
        {"email": "test@example.com", "password": ""},
        {"email": "test@example.com"},
        {"password": "Password123"},
    ],
)
def test_login_invalid_credentials(
    client, mock_auth_service, payload
):
    response = client.post(
        "/api/auth/login",
        json=payload,
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"
    assert "obligatorios" in data["message"]

    mock_auth_service.login_user.assert_not_called()


def test_login_service_error(client, mock_auth_service):
    mock_auth_service.login_user.side_effect = Exception(
        "Credenciales incorrectas"
    )

    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "incorrecta",
        },
    )

    data = response.get_json()

    assert response.status_code == 401
    assert data["status"] == "error"
    assert "Credenciales incorrectas" in data["message"]


def test_register_success(client, mock_auth_service):
    mock_auth_service.register_user.return_value = SimpleNamespace(
        user=SimpleNamespace(
            id="new-user-123",
            email="new@example.com",
        ),
    )

    response = client.post(
        "/api/auth/register",
        json={
            "email": "new@example.com",
            "password": "Password123",
        },
    )

    data = response.get_json()

    assert response.status_code == 201
    assert data["status"] == "success"
    assert data["user"]["id"] == "new-user-123"
    assert data["user"]["email"] == "new@example.com"

    mock_auth_service.register_user.assert_called_once_with(
        "new@example.com",
        "Password123",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"email": "", "password": "Password123"},
        {"email": "new@example.com", "password": ""},
        {"email": "new@example.com"},
        {"password": "Password123"},
    ],
)
def test_register_invalid_credentials(
    client, mock_auth_service, payload
):
    response = client.post(
        "/api/auth/register",
        json=payload,
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"
    assert "obligatorios" in data["message"]

    mock_auth_service.register_user.assert_not_called()


def test_register_service_error(client, mock_auth_service):
    mock_auth_service.register_user.side_effect = Exception(
        "El usuario ya existe"
    )

    response = client.post(
        "/api/auth/register",
        json={
            "email": "existing@example.com",
            "password": "Password123",
        },
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"
    assert "El usuario ya existe" in data["message"]
