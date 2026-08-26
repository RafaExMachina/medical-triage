"""HTTP routes exposed by the medical classification API."""

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from medical_triage.application.use_cases import (
    ClassifyMedicalTextUseCase,
)
from medical_triage.observability.metrics import PREDICTIONS_TOTAL
from medical_triage.presentation.api.dependencies import (
    get_classification_use_case,
)
from medical_triage.presentation.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    HealthResponse,
)

router = APIRouter()


@router.get(
    "/metrics",
    include_in_schema=False,
    response_class=Response,
)
def metrics() -> Response:
    """Expose application metrics in Prometheus format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health() -> HealthResponse:
    """Return the current health status of the API.

    Returns:
        HealthResponse indicating that the service is operational.
    """
    return HealthResponse(status="healthy")


@router.post(
    "/predict",
    response_model=ClassificationResponse,
    tags=["Prediction"],
)
def predict(
    request: ClassificationRequest,
    use_case: ClassifyMedicalTextUseCase = Depends(get_classification_use_case),
) -> ClassificationResponse:
    """Classify a medical text using the loaded NLP model.

    Args:
        request: Validated HTTP request containing medical text.
        use_case: Injected medical-text classification use case.

    Returns:
        ClassificationResponse containing the predicted category,
        model metadata, confidence, and inference latency.

    Raises:
        HTTPException: If the application cannot process the supplied
            medical text.
    """
    start_time = perf_counter()

    try:
        result = use_case.execute(request.text)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    inference_ms = (perf_counter() - start_time) * 1000

    PREDICTIONS_TOTAL.labels(
        label_name=result.label_name,
    ).inc()

    return ClassificationResponse(
        label_id=result.label_id,
        label_name=result.label_name,
        confidence=result.confidence,
        model_version=result.model_version,
        inference_ms=inference_ms,
    )
