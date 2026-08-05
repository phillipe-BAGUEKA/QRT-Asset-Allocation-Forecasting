from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import EXPECTED_FEATURE_COLUMNS
from app.main import create_app
from app.model_service import QRTModelService


def test_model_info_returns_success_and_serializable_json(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/model-info")

    assert response.status_code == 200
    assert json.loads(json.dumps(response.json())) == response.json()


def test_model_info_returns_feature_order_and_valid_threshold(
    api_test_client: TestClient,
) -> None:
    model_info = api_test_client.get("/model-info").json()

    assert model_info["features"] == list(EXPECTED_FEATURE_COLUMNS)
    assert model_info["feature_count"] == 20
    assert 0.0 <= model_info["threshold"] <= 1.0
    assert model_info["model_version"] == "v1"


def test_model_info_returns_available_training_metadata(
    api_test_client: TestClient,
) -> None:
    model_info = api_test_client.get("/model-info").json()

    assert model_info["training_period"] == {
        "start": "DATE_0001",
        "end": "DATE_2522",
    }
    assert model_info["training_observations"] == 527073
    assert model_info["reference_metrics"]["mean_valid_roc_auc"] == 0.52936


def test_model_info_returns_503_when_metadata_is_missing(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    metadata_path = tmp_path / "missing.json"
    application = create_app(
        service_loader=lambda: QRTModelService.load(
            model_path=model_path,
            metadata_path=metadata_path,
        )
    )

    with TestClient(application) as client:
        response = client.get("/model-info")

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_model_info_returns_503_when_metadata_json_is_invalid(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("not-json", encoding="utf-8")
    application = create_app(
        service_loader=lambda: QRTModelService.load(
            model_path=model_path,
            metadata_path=metadata_path,
        )
    )

    with TestClient(application) as client:
        response = client.get("/model-info")

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
