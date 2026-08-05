from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from app.config import EXPECTED_FEATURE_COLUMNS
from app.model_service import (
    MetadataValidationError,
    ModelUnavailableError,
    QRTModelService,
)
from tests.api_fakes import FakePipeline


def test_service_converts_none_to_nan_and_preserves_feature_order(
    model_service: QRTModelService,
    fake_pipeline: FakePipeline,
) -> None:
    values = {feature: 0.0 for feature in EXPECTED_FEATURE_COLUMNS}
    values["RET_7"] = None

    result = model_service.predict(values)

    frame = fake_pipeline.received_frames[0]
    assert frame.columns.tolist() == list(EXPECTED_FEATURE_COLUMNS)
    assert frame.shape == (1, 20)
    assert np.isnan(frame.loc[0, "RET_7"])
    assert result["positive_probability"] == pytest.approx(0.8)


def test_service_uses_positive_class_index_even_when_first(
    valid_metadata: dict[str, Any],
) -> None:
    pipeline = FakePipeline(classes=(1, 0), probabilities=(0.73, 0.27))
    service = QRTModelService(pipeline, valid_metadata)

    result = service.predict(
        {feature: 0.0 for feature in EXPECTED_FEATURE_COLUMNS}
    )

    assert service.positive_class_index == 0
    assert result["positive_probability"] == pytest.approx(0.73)
    assert result["predicted_class"] == 1


@pytest.mark.parametrize(
    ("metadata_update", "message"),
    [
        ({"feature_columns": ["RET_1"]}, "RET_1 through RET_20"),
        ({"feature_count": 19}, "feature_count"),
        ({"prediction_threshold": -0.1}, "prediction_threshold"),
        ({"prediction_threshold": float("nan")}, "prediction_threshold"),
        ({"model_name": ""}, "model_name"),
    ],
)
def test_service_rejects_invalid_metadata(
    fake_pipeline: FakePipeline,
    valid_metadata: dict[str, Any],
    metadata_update: dict[str, Any],
    message: str,
) -> None:
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata.update(metadata_update)

    with pytest.raises(MetadataValidationError, match=message):
        QRTModelService(fake_pipeline, invalid_metadata)


def test_service_rejects_pipeline_without_predict_proba(
    valid_metadata: dict[str, Any],
) -> None:
    pipeline = Mock(classes_=np.asarray([0, 1]), n_features_in_=20)
    pipeline.predict_proba = None

    with pytest.raises(ModelUnavailableError, match="predict_proba"):
        QRTModelService(pipeline, valid_metadata)


def test_service_rejects_pipeline_without_positive_class(
    valid_metadata: dict[str, Any],
) -> None:
    pipeline = FakePipeline(classes=(0, 2), probabilities=(0.4, 0.6))

    with pytest.raises(ModelUnavailableError, match="positive class 1"):
        QRTModelService(pipeline, valid_metadata)


def test_load_uses_joblib_once_with_valid_local_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_pipeline: FakePipeline,
    valid_metadata: dict[str, Any],
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(valid_metadata), encoding="utf-8")
    load_mock = Mock(return_value=fake_pipeline)
    monkeypatch.setattr("app.model_service.joblib.load", load_mock)

    service = QRTModelService.load(model_path, metadata_path)

    assert service.pipeline is fake_pipeline
    load_mock.assert_called_once_with(model_path)


def test_load_rejects_missing_model_without_calling_joblib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_mock = Mock()
    monkeypatch.setattr("app.model_service.joblib.load", load_mock)

    with pytest.raises(ModelUnavailableError, match="artifact is unavailable"):
        QRTModelService.load(
            tmp_path / "missing.joblib",
            tmp_path / "missing.json",
        )

    load_mock.assert_not_called()
