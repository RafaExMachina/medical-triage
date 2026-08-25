"""Unit tests for the medical text classification use case."""

import pytest

from medical_triage.application.use_cases import (
    ClassifyMedicalTextUseCase,
)
from medical_triage.domain.entities import (
    ClassificationResult,
)


class FakeClassifier:
    """Fake classifier used to isolate the application layer during tests."""

    def predict(
        self,
        text: str,
    ) -> ClassificationResult:
        """Return a deterministic fake classification.

        Args:
            text: Medical text received by the fake classifier.

        Returns:
            Fixed ClassificationResult for testing purposes.
        """
        return ClassificationResult(
            label_id=4,
            label_name="Cardiovascular diseases",
            confidence=0.95,
            model_version="fake-v1",
        )


def test_classify_medical_text() -> None:
    """Verify that a valid text is correctly forwarded to the classifier."""
    classifier = FakeClassifier()

    use_case = ClassifyMedicalTextUseCase(classifier=classifier)

    result = use_case.execute("Patient presenting acute cardiovascular symptoms.")

    assert result.label_id == 4
    assert result.label_name == "Cardiovascular diseases"
    assert result.confidence == 0.95
    assert result.model_version == "fake-v1"


def test_empty_medical_text_should_raise_error() -> None:
    """Verify that an empty medical text raises ValueError."""
    classifier = FakeClassifier()

    use_case = ClassifyMedicalTextUseCase(classifier=classifier)

    with pytest.raises(
        ValueError,
        match="Medical text cannot be empty",
    ):
        use_case.execute("")


def test_short_medical_text_should_raise_error() -> None:
    """Verify that medical texts shorter than the minimum are rejected."""
    classifier = FakeClassifier()

    use_case = ClassifyMedicalTextUseCase(classifier=classifier)

    with pytest.raises(
        ValueError,
        match="at least 20 characters",
    ):
        use_case.execute("Chest pain.")
