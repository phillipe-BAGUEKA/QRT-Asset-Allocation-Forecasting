from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from app.config import PROJECT_ROOT
from frontend import api_client


@pytest.fixture(autouse=True)
def configure_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRT_API_BASE_URL", "http://api.test:9000/")


def _response(status_code: int, payload: object) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.json.return_value = payload
    return response


def test_get_health_returns_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_mock = Mock(return_value=_response(200, {"status": "healthy"}))
    monkeypatch.setattr(api_client.requests, "request", request_mock)

    result = api_client.get_health(timeout=2.0)

    assert result == {"status": "healthy"}
    request_mock.assert_called_once_with(
        method="GET",
        url="http://api.test:9000/health",
        json=None,
        timeout=2.0,
    )


def test_get_model_info_returns_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"model_name": "gradient_boosting_ret20_v1"}
    monkeypatch.setattr(
        api_client.requests,
        "request",
        Mock(return_value=_response(200, payload)),
    )

    assert api_client.get_model_info() == payload


def test_predict_sends_payload_and_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payload = {"RET_1": None}
    response_payload = {"positive_probability": 0.53}
    request_mock = Mock(return_value=_response(200, response_payload))
    monkeypatch.setattr(api_client.requests, "request", request_mock)

    assert api_client.predict(request_payload) == response_payload
    assert request_mock.call_args.kwargs["method"] == "POST"
    assert request_mock.call_args.kwargs["json"] == request_payload


def test_connection_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "request",
        Mock(side_effect=requests.ConnectionError("offline")),
    )

    with pytest.raises(api_client.ApiConnectionError, match="unreachable"):
        api_client.get_health()


def test_timeout_is_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "request",
        Mock(side_effect=requests.Timeout("late")),
    )

    with pytest.raises(api_client.ApiTimeoutError, match="10 seconds"):
        api_client.get_health()


@pytest.mark.parametrize("status_code", [422, 500])
def test_http_error_preserves_status_code(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "request",
        Mock(return_value=_response(status_code, {"detail": "failure"})),
    )

    with pytest.raises(api_client.ApiResponseError) as error:
        api_client.get_model_info()

    assert error.value.status_code == status_code
    assert error.value.detail == "failure"


def test_non_json_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(200, {})
    response.json.side_effect = ValueError("invalid json")
    monkeypatch.setattr(
        api_client.requests,
        "request",
        Mock(return_value=response),
    )

    with pytest.raises(api_client.ApiInvalidResponseError, match="valid JSON"):
        api_client.get_health()


def test_streamlit_pages_import_without_running_navigation() -> None:
    from frontend import streamlit_app
    from frontend.views import api_overview, home, model_info, prediction

    assert callable(streamlit_app.main)
    assert callable(home.render)
    assert callable(prediction.render)
    assert callable(model_info.render)
    assert callable(api_overview.render)


def test_streamlit_application_renders_without_navigation_error() -> None:
    from streamlit.testing.v1 import AppTest

    application = AppTest.from_file(
        PROJECT_ROOT / "frontend" / "streamlit_app.py",
        default_timeout=15,
    ).run()

    assert not application.exception
