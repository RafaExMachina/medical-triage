"""Ports that define contracts required by the application layer."""

from typing import Protocol

from medical_triage.domain.entities import ClassificationResult


class ClassifierPort(Protocol):
    """Define the contract expected from a text classification service.

    Implementations may use Scikit-learn, ONNX Runtime, PyTorch,
    Transformers, or any other inference technology.
    """

    def predict(self, text: str) -> ClassificationResult:
        """Classify a medical text.

        Args:
            text: Medical text to be classified.

        Returns:
            ClassificationResult containing the predicted class,
            confidence score, and model metadata.
        """
        ...
