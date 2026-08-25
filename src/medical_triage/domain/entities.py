"""Domain entities used by the medical text classification application."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Represent the result produced by a medical text classifier.

    Attributes:
        label_id: Numeric identifier of the predicted class.
        label_name: Human-readable name of the predicted class.
        confidence: Prediction confidence between 0 and 1, when available.
        model_version: Identifier of the model version used for inference.
    """

    label_id: int
    label_name: str
    confidence: float | None
    model_version: str
