"""Pydantic request and response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, create_model

from app.config import EXPECTED_FEATURE_COLUMNS


PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature_name: (FiniteFloat | None, Field(...))
        for feature_name in EXPECTED_FEATURE_COLUMNS
    },
)


class PredictionResponse(BaseModel):
    """Response returned for one allocation prediction."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str | None
    positive_probability: float = Field(ge=0.0, le=1.0)
    predicted_class: Literal[0, 1]
    threshold: float = Field(ge=0.0, le=1.0)
