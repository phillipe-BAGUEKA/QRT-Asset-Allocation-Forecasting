"""Loading, validation, and inference for the frozen QRT pipeline."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import (
    EXPECTED_FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
)


class ModelServiceError(RuntimeError):
    """Base exception for model loading and inference failures."""


class ModelUnavailableError(ModelServiceError):
    """Raised when a required model resource cannot be loaded."""


class MetadataValidationError(ModelServiceError):
    """Raised when persisted metadata does not satisfy the API contract."""


class PredictionError(ModelServiceError):
    """Raised when the loaded pipeline cannot produce a valid prediction."""


class QRTModelService:
    """Validated in-memory interface to the frozen QRT model pipeline."""

    def __init__(self, pipeline: Any, metadata: Mapping[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = dict(metadata)
        self.feature_columns = self._validate_metadata(self.metadata)
        self.threshold = self._extract_threshold(self.metadata)
        self.model_name = self._extract_model_name(self.metadata)
        self.model_version = self._extract_model_version(self.metadata)
        self.positive_class_index = self._validate_pipeline(pipeline)

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> "QRTModelService":
        """Load and validate persisted resources once during API startup."""
        if not model_path.is_file():
            raise ModelUnavailableError(
                f"Model artifact is unavailable: {model_path}"
            )
        if not metadata_path.is_file():
            raise ModelUnavailableError(
                f"Model metadata is unavailable: {metadata_path}"
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except json.JSONDecodeError as exc:
            raise MetadataValidationError(
                f"Model metadata is not valid JSON: {metadata_path}"
            ) from exc
        except OSError as exc:
            raise ModelUnavailableError(
                f"Model metadata could not be read: {metadata_path}"
            ) from exc

        if not isinstance(metadata, dict):
            raise MetadataValidationError(
                "Model metadata must contain a JSON object."
            )

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Model artifact could not be loaded: {model_path}"
            ) from exc

        return cls(pipeline=pipeline, metadata=metadata)

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
        feature_columns = metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not all(
            isinstance(feature, str) for feature in feature_columns
        ):
            raise MetadataValidationError(
                "Metadata field 'feature_columns' must be a list of strings."
            )

        expected_features = list(EXPECTED_FEATURE_COLUMNS)
        if feature_columns != expected_features:
            raise MetadataValidationError(
                "Metadata features must be exactly RET_1 through RET_20 in "
                "ascending order."
            )

        feature_count = metadata.get("feature_count")
        if isinstance(feature_count, bool) or feature_count != len(
            expected_features
        ):
            raise MetadataValidationError(
                "Metadata field 'feature_count' must equal 20."
            )

        return tuple(feature_columns)

    @staticmethod
    def _extract_threshold(metadata: Mapping[str, Any]) -> float:
        threshold = metadata.get("prediction_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise MetadataValidationError(
                "Metadata field 'prediction_threshold' must be finite and "
                "between 0 and 1."
            )
        return float(threshold)

    @staticmethod
    def _extract_model_name(metadata: Mapping[str, Any]) -> str:
        model_name = metadata.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise MetadataValidationError(
                "Metadata field 'model_name' must be a non-empty string."
            )
        return model_name

    @staticmethod
    def _extract_model_version(metadata: Mapping[str, Any]) -> str | None:
        explicit_version = metadata.get("model_version")
        if isinstance(explicit_version, str) and explicit_version.strip():
            return explicit_version

        model_name = metadata.get("model_name")
        if isinstance(model_name, str):
            version_match = re.search(r"_(v\d+)$", model_name)
            if version_match:
                return version_match.group(1)
        return None

    def _validate_pipeline(self, pipeline: Any) -> int:
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise ModelUnavailableError(
                "Loaded model does not expose predict_proba()."
            )

        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            raise ModelUnavailableError(
                "Loaded model does not expose fitted classes_."
            )

        class_values = np.asarray(classes)
        if class_values.ndim != 1:
            raise ModelUnavailableError(
                "Loaded model classes_ must be one-dimensional."
            )

        positive_indices = np.flatnonzero(class_values == 1)
        if positive_indices.size != 1:
            raise ModelUnavailableError(
                "Loaded model classes_ must contain the positive class 1 "
                "exactly once."
            )

        n_features = getattr(pipeline, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(
            self.feature_columns
        ):
            raise ModelUnavailableError(
                "Loaded model expects a different number of features."
            )

        feature_names = getattr(pipeline, "feature_names_in_", None)
        if feature_names is not None and list(feature_names) != list(
            self.feature_columns
        ):
            raise ModelUnavailableError(
                "Loaded model feature names do not match the metadata order."
            )

        return int(positive_indices[0])

    def get_model_info(self) -> dict[str, Any]:
        """Return a serializable public view of available model metadata."""
        software_versions = {
            response_key: self.metadata[metadata_key]
            for response_key, metadata_key in (
                ("python", "python_version"),
                ("scikit_learn", "scikit_learn_version"),
                ("joblib", "joblib_version"),
                ("numpy", "numpy_version"),
                ("pandas", "pandas_version"),
            )
            if metadata_key in self.metadata
        }

        model_info: dict[str, Any] = {
            "model_name": self.model_name,
            "features": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "threshold": self.threshold,
        }

        optional_values = {
            "model_version": self.model_version,
            "model_type": self.metadata.get("model_family"),
            "pipeline_steps": self.metadata.get("pipeline"),
            "training_observations": self.metadata.get(
                "training_row_count"
            ),
            "training_dates": self.metadata.get("training_date_count"),
            "reference_metrics": self.metadata.get("validation_reference"),
            "software_versions": software_versions or None,
            "model_sha256": self.metadata.get("artifact_sha256"),
        }
        model_info.update(
            {
                key: value
                for key, value in optional_values.items()
                if value is not None
            }
        )

        training_start = self.metadata.get("training_start_date")
        training_end = self.metadata.get("training_end_date")
        if training_start is not None or training_end is not None:
            model_info["training_period"] = {
                "start": training_start,
                "end": training_end,
            }

        return model_info

    def predict(self, values: Mapping[str, float | None]) -> dict[str, Any]:
        """Predict the positive-class probability for one validated row."""
        row = [
            np.nan if values[feature] is None else float(values[feature])
            for feature in self.feature_columns
        ]
        feature_frame = pd.DataFrame([row], columns=self.feature_columns)

        try:
            probabilities = np.asarray(
                self.pipeline.predict_proba(feature_frame), dtype=float
            )
        except Exception as exc:
            raise PredictionError("Model inference failed.") from exc

        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != 1
            or self.positive_class_index >= probabilities.shape[1]
        ):
            raise PredictionError(
                "Model returned an invalid probability array."
            )

        positive_probability = float(
            probabilities[0, self.positive_class_index]
        )
        if not math.isfinite(positive_probability) or not (
            0.0 <= positive_probability <= 1.0
        ):
            raise PredictionError(
                "Model returned an invalid positive-class probability."
            )

        predicted_class = int(positive_probability >= self.threshold)
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "positive_probability": positive_probability,
            "predicted_class": predicted_class,
            "threshold": self.threshold,
        }
