from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_success_status() -> None:
    response = client.get("/")

    assert response.status_code == 200


def test_root_returns_service_information() -> None:
    response = client.get("/")

    assert response.json() == {
        "name": "QRT Prediction API",
        "message": "QRT model inference service",
        "docs": "/docs",
    }


def test_health_returns_success_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.json() == {"status": "healthy"}


def test_unknown_route_returns_not_found() -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404


def test_health_rejects_post_requests() -> None:
    response = client.post("/health")

    assert response.status_code == 405
