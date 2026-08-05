from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import EXPECTED_FEATURE_COLUMNS
from app.main import create_app
from app.model_service import QRTModelService
from tests.api_fakes import FakePipeline


@pytest.fixture
def valid_metadata() -> dict[str, Any]:
    return {
        "artifact_sha256": "abc123",
        "feature_columns": list(EXPECTED_FEATURE_COLUMNS),
        "feature_count": len(EXPECTED_FEATURE_COLUMNS),
        "joblib_version": "1.5.3",
        "model_family": "GradientBoostingClassifier",
        "model_name": "gradient_boosting_ret20_v1",
        "numpy_version": "2.4.4",
        "pandas_version": "2.3.3",
        "pipeline": ["SimpleImputer", "GradientBoostingClassifier"],
        "prediction_threshold": 0.5,
        "python_version": "3.13.7",
        "scikit_learn_version": "1.8.0",
        "training_date_count": 2522,
        "training_end_date": "DATE_2522",
        "training_row_count": 527073,
        "training_start_date": "DATE_0001",
        "validation_reference": {"mean_valid_roc_auc": 0.52936},
    }


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def model_service(
    fake_pipeline: FakePipeline,
    valid_metadata: dict[str, Any],
) -> QRTModelService:
    return QRTModelService(fake_pipeline, valid_metadata)


@pytest.fixture
def inference_app(model_service: QRTModelService) -> FastAPI:
    return create_app(service_loader=lambda: model_service)


@pytest.fixture
def api_test_client(inference_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(inference_app) as client:
        yield client
