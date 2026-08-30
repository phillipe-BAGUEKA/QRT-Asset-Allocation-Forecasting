"""Centralized paths and environment settings for the inference application."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_ret20_final.joblib"
METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_ret20_final.manifest.json"
)
IMAGES_DIR = PROJECT_ROOT / "images"
QRT_LOGO_PATH = IMAGES_DIR / "QRT-Brand-Master-Full-WO-HR.png"

EXPECTED_FEATURE_COLUMNS = tuple(
    f"RET_{index}" for index in range(1, 21)
)

API_BASE_URL_ENV_VAR = "QRT_API_BASE_URL"
DEFAULT_API_BASE_URL = "http://localhost:8000"


def get_api_base_url() -> str:
    """Return the normalized API base URL configured for the frontend."""
    configured_url = os.getenv(API_BASE_URL_ENV_VAR, DEFAULT_API_BASE_URL)
    normalized_url = configured_url.strip().rstrip("/")
    return normalized_url or DEFAULT_API_BASE_URL
