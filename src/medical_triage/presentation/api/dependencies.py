"""Dependency providers used by the FastAPI application."""

from functools import lru_cache

from medical_triage.application.use_cases import (
    ClassifyMedicalTextUseCase,
)
from medical_triage.config import get_settings
from medical_triage.infrastructure.sklearn_classifier import (
    SklearnClassifierAdapter,
)


@lru_cache
def get_classifier() -> SklearnClassifierAdapter:
    """Load and cache the machine-learning classifier.

    The classifier is loaded only once during the application lifecycle,
    avoiding repeated model loading for every HTTP request.

    Returns:
        Cached Scikit-learn classifier adapter.
    """
    settings = get_settings()

    return SklearnClassifierAdapter.from_file(settings.model_path)


@lru_cache
def get_classification_use_case() -> ClassifyMedicalTextUseCase:
    """Create and cache the medical classification use case.

    Returns:
        Configured ClassifyMedicalTextUseCase instance.
    """
    classifier = get_classifier()

    return ClassifyMedicalTextUseCase(classifier=classifier)
