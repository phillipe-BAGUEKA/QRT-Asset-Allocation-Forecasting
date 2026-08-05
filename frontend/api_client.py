"""HTTP client used by all Streamlit views."""

from __future__ import annotations

from typing import Any

import requests

from app.config import get_api_base_url


DEFAULT_TIMEOUT_SECONDS = 10.0


class ApiClientError(RuntimeError):
    """Base exception for clean frontend API failures."""


class ApiConnectionError(ApiClientError):
    """Raised when the FastAPI service cannot be reached."""


class ApiTimeoutError(ApiClientError):
    """Raised when the FastAPI service does not respond in time."""


class ApiResponseError(ApiClientError):
    """Raised when FastAPI returns a non-successful HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API returned HTTP {status_code}: {detail}")


class ApiInvalidResponseError(ApiClientError):
    """Raised when FastAPI does not return the expected JSON object."""


def _request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    url = f"{get_api_base_url()}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise ApiTimeoutError(
            f"The API did not respond within {timeout:g} seconds."
        ) from exc
    except requests.ConnectionError as exc:
        raise ApiConnectionError(
            f"The API is unreachable at {get_api_base_url()}."
        ) from exc
    except requests.RequestException as exc:
        raise ApiClientError("The API request failed.") from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise ApiInvalidResponseError(
            "The API response is not valid JSON."
        ) from exc

    if not response.ok:
        detail = (
            response_payload.get("detail", "The API returned an error.")
            if isinstance(response_payload, dict)
            else "The API returned an error."
        )
        raise ApiResponseError(response.status_code, str(detail))

    if not isinstance(response_payload, dict):
        raise ApiInvalidResponseError(
            "The API response must be a JSON object."
        )
    return response_payload


def get_health(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Return the FastAPI liveness response."""
    return _request_json("GET", "/health", timeout=timeout)


def get_model_info(
    *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Return the public metadata exposed by FastAPI."""
    return _request_json("GET", "/model-info", timeout=timeout)


def predict(
    payload: dict[str, float | None],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Request one prediction from FastAPI."""
    return _request_json("POST", "/predict", payload=payload, timeout=timeout)
