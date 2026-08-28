"""Unit tests for API dependency providers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from medical_triage.domain.entities import (
    ClassificationResult,
)
from medical_triage.presentation.api import (
    dependencies,
)


class FakeClassifier:
    """Provide a minimal classifier for dependency tests."""

    def predict(
        self,
        text: str,
    ) -> ClassificationResult:
        """Return a deterministic classification."""
        return ClassificationResult(
            label_id=4,
            label_name="cardiovascular diseases",
            confidence=0.90,
            model_version="fake-model",
        )


@pytest.fixture(autouse=True)
def clear_dependency_caches() -> Iterator[None]:
    """Clear cached dependencies around every test."""
    dependencies.get_classifier.cache_clear()
    dependencies.get_classification_use_case.cache_clear()

    yield

    dependencies.get_classifier.cache_clear()
    dependencies.get_classification_use_case.cache_clear()


def test_get_classifier_should_select_onnx_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that MODEL_BACKEND=onnx selects the ONNX adapter."""
    fake_classifier = FakeClassifier()

    monkeypatch.setenv(
        "MODEL_BACKEND",
        "onnx",
    )

    monkeypatch.setattr(
        dependencies.OnnxClassifierAdapter,
        "from_files",
        classmethod(lambda _cls, **_kwargs: fake_classifier),
    )

    classifier = dependencies.get_classifier()

    assert classifier is fake_classifier


def test_get_classifier_should_select_sklearn_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that MODEL_BACKEND=sklearn selects the sklearn adapter."""
    fake_classifier = FakeClassifier()

    monkeypatch.setenv(
        "MODEL_BACKEND",
        "sklearn",
    )

    monkeypatch.setattr(
        dependencies.SklearnClassifierAdapter,
        "from_file",
        classmethod(lambda _cls, _path: fake_classifier),
    )

    classifier = dependencies.get_classifier()

    assert classifier is fake_classifier


def test_get_classifier_should_default_to_onnx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that ONNX is selected when MODEL_BACKEND is absent."""
    fake_classifier = FakeClassifier()

    monkeypatch.delenv(
        "MODEL_BACKEND",
        raising=False,
    )

    monkeypatch.setattr(
        dependencies.OnnxClassifierAdapter,
        "from_files",
        classmethod(lambda _cls, **_kwargs: fake_classifier),
    )

    classifier = dependencies.get_classifier()

    assert classifier is fake_classifier
