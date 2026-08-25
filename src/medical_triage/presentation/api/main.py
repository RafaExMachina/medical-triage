"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from medical_triage.config import get_settings
from medical_triage.observability.logging import (
    configure_logging,
    get_logger,
)
from medical_triage.presentation.api.dependencies import (
    get_classification_use_case,
)
from medical_triage.presentation.api.routes import router

configure_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources.

    The machine-learning model is loaded during application startup
    to avoid loading the artifact for every inference request.

    Args:
        _app: FastAPI application instance.

    Yields:
        Control back to FastAPI while the application is running.
    """
    logger.info("Starting Medical Triage API")

    get_classification_use_case()

    logger.info("Machine learning model loaded successfully")

    yield

    logger.info("Shutting down Medical Triage API")


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for medical text classification.",
    lifespan=lifespan,
)

app.include_router(router)
