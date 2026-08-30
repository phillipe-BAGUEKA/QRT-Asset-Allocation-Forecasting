"""FastAPI application for the QRT inference service."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model_service import (
    ModelServiceError,
    PredictionError,
    QRTModelService,
)
from app.schemas import PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)
ServiceLoader = Callable[[], QRTModelService]


def _json_safe_validation_detail(value: Any) -> Any:
    """Replace non-finite floats in validation errors with readable text."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {
            key: _json_safe_validation_detail(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_detail(item) for item in value]
    return value


def _get_model_service(request: Request) -> QRTModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction model is unavailable. Check the API startup "
                "logs and local model resources."
            ),
        )
    return service


def create_app(
    service_loader: ServiceLoader = QRTModelService.load,
) -> FastAPI:
    """Create an application with one model-loading lifespan."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.model_service = service_loader()
            application.state.model_startup_error = None
        except Exception as exc:
            application.state.model_service = None
            application.state.model_startup_error = str(exc)
            LOGGER.exception("The QRT model service could not start.")
        yield
        application.state.model_service = None

    application = FastAPI(
        title="QRT Prediction API",
        description=(
            "Inference API for the QRT asset allocation forecasting project."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": _json_safe_validation_detail(exc.errors())
            },
        )

    @application.get("/")
    def read_root() -> dict[str, str]:
        """Return basic service information."""
        return {
            "name": "QRT Prediction API",
            "message": "QRT model inference service",
            "docs": "/docs",
        }

    @application.get("/health")
    def read_health(request: Request) -> dict[str, Any]:
        """Report process, model and schema readiness."""
        service = getattr(request.app.state, "model_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The prediction model is unavailable.",
            )
        return {
            "status": "healthy",
            "model_loaded": True,
            "schema_available": len(service.feature_columns) == 20,
            "model_version": service.model_version,
        }

    @application.get("/model-info")
    def read_model_info(
        service: QRTModelService = Depends(_get_model_service),
    ) -> dict[str, Any]:
        """Return the public, serializable metadata available for the model."""
        return service.get_model_info()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        request_body: PredictionRequest,
        service: QRTModelService = Depends(_get_model_service),
    ) -> PredictionResponse:
        """Predict future positive performance for one allocation row."""
        try:
            result = service.predict(request_body.model_dump())
        except PredictionError as exc:
            LOGGER.exception("The QRT pipeline failed during prediction.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The prediction could not be completed.",
            ) from exc
        except ModelServiceError as exc:
            LOGGER.exception("The QRT model service is unavailable.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The prediction model is unavailable.",
            ) from exc
        return PredictionResponse(**result)

    return application


app = create_app()
