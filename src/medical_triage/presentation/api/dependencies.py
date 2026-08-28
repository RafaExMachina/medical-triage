"""Dependency providers used by the FastAPI application."""

from functools import lru_cache

from medical_triage.application.use_cases import (
    ClassifyMedicalTextUseCase,
)
from medical_triage.config import get_settings
from medical_triage.domain.ports import ClassifierPort
from medical_triage.infrastructure.onnx_classifier import (
    OnnxClassifierAdapter,
)
from medical_triage.infrastructure.sklearn_classifier import (
    SklearnClassifierAdapter,
)


@lru_cache
def get_classifier() -> ClassifierPort:
    """Load and cache the configured machine-learning classifier.

    The inference backend is selected through MODEL_BACKEND.

    Supported values:

    - ``onnx``: optimized ONNX Runtime pipeline.
    - ``sklearn``: original Scikit-learn pipeline.

    Returns:
        Cached classifier implementing ClassifierPort.
    """
    settings = get_settings()

    if settings.model_backend == "onnx":
        return OnnxClassifierAdapter.from_files(
            model_path=settings.onnx_model_path,
            metadata_path=settings.onnx_metadata_path,
        )

    return SklearnClassifierAdapter.from_file(
        settings.model_path,
    )


@lru_cache
def get_classification_use_case() -> ClassifyMedicalTextUseCase:
    """Create and cache the medical classification use case.

    Returns:
        Configured ClassifyMedicalTextUseCase instance.
    """
    classifier = get_classifier()

    return ClassifyMedicalTextUseCase(
        classifier=classifier,
    )
