from fastapi.testclient import TestClient

def test_root_returns_success_status(api_test_client: TestClient) -> None:
    response = api_test_client.get("/")

    assert response.status_code == 200


def test_root_returns_service_information(api_test_client: TestClient) -> None:
    response = api_test_client.get("/")

    assert response.json() == {
        "name": "QRT Prediction API",
        "message": "QRT model inference service",
        "docs": "/docs",
    }


def test_health_returns_success_status(api_test_client: TestClient) -> None:
    response = api_test_client.get("/health")

    assert response.status_code == 200


def test_health_returns_healthy_status(api_test_client: TestClient) -> None:
    response = api_test_client.get("/health")

    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "schema_available": True,
        "model_version": "2.0.0",
    }


def test_unknown_route_returns_not_found(api_test_client: TestClient) -> None:
    response = api_test_client.get("/does-not-exist")

    assert response.status_code == 404


def test_health_rejects_post_requests(api_test_client: TestClient) -> None:
    response = api_test_client.post("/health")

    assert response.status_code == 405
