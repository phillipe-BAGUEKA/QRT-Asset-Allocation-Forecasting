from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import EXPECTED_FEATURE_COLUMNS
from app.main import create_app
from app.model_service import ModelUnavailableError
from tests.api_fakes import FakePipeline


def _valid_payload(value: float | None = 0.01) -> dict[str, float | None]:
    return {feature: value for feature in EXPECTED_FEATURE_COLUMNS}


def test_predict_returns_probability_and_threshold_consistent_class(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    result = response.json()
    assert 0.0 <= result["positive_probability"] <= 1.0
    assert result["positive_probability"] == pytest.approx(0.8)
    assert result["predicted_class"] == int(
        result["positive_probability"] >= result["threshold"]
    )
    assert result["model_name"] == "gradient_boosting_ret20_v1"
    assert result["model_version"] == "v1"


def test_predict_builds_one_row_with_exact_feature_order(
    api_test_client: TestClient,
    fake_pipeline: FakePipeline,
) -> None:
    response = api_test_client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    assert len(fake_pipeline.received_frames) == 1
    frame = fake_pipeline.received_frames[0]
    assert frame.shape == (1, len(EXPECTED_FEATURE_COLUMNS))
    assert frame.columns.tolist() == list(EXPECTED_FEATURE_COLUMNS)


def test_predict_accepts_null_and_passes_nan_to_pipeline(
    api_test_client: TestClient,
    fake_pipeline: FakePipeline,
) -> None:
    payload = _valid_payload()
    payload["RET_3"] = None

    response = api_test_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert np.isnan(fake_pipeline.received_frames[0].loc[0, "RET_3"])


def test_predict_rejects_missing_field(api_test_client: TestClient) -> None:
    payload = _valid_payload()
    del payload["RET_20"]

    response = api_test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_wrong_type(api_test_client: TestClient) -> None:
    payload: dict[str, Any] = _valid_payload()
    payload["RET_1"] = "not-a-number"

    response = api_test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_extra_field(api_test_client: TestClient) -> None:
    payload: dict[str, Any] = _valid_payload()
    payload["UNEXPECTED"] = 1.0

    response = api_test_client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_predict_rejects_non_finite_values(
    api_test_client: TestClient,
    invalid_value: float,
) -> None:
    payload = _valid_payload()
    payload["RET_1"] = invalid_value

    response = api_test_client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_predict_rejects_get_method(api_test_client: TestClient) -> None:
    response = api_test_client.get("/predict")

    assert response.status_code == 405


def test_predict_returns_503_when_model_is_unavailable() -> None:
    def unavailable_loader():
        raise ModelUnavailableError("missing model")

    application = create_app(service_loader=unavailable_loader)
    with TestClient(application) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_predict_finds_positive_class_from_classes_not_column_position(
    api_test_client: TestClient,
    fake_pipeline: FakePipeline,
) -> None:
    assert fake_pipeline.classes_.tolist() == [1, 0]

    response = api_test_client.post("/predict", json=_valid_payload())

    assert response.json()["positive_probability"] == pytest.approx(0.8)
