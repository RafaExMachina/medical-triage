"""Scikit-learn adapter for medical text classification."""

from pathlib import Path
from typing import Any

import joblib

from medical_triage.domain.entities import ClassificationResult


class SklearnClassifierAdapter:
    """Adapt a Scikit-learn model to the ClassifierPort contract.

    This adapter isolates Scikit-learn-specific implementation details
    from the application and domain layers.
    """

    def __init__(
        self,
        model: Any,
        labels: dict[int, str],
        model_version: str,
    ) -> None:
        """Initialize the Scikit-learn classifier adapter.

        Args:
            model: Trained Scikit-learn model or pipeline.
            labels: Mapping between numeric labels and readable names.
            model_version: Identifier of the loaded model.
        """
        self._model = model
        self._labels = labels
        self._model_version = model_version

    @classmethod
    def from_file(
        cls,
        model_path: Path,
    ) -> "SklearnClassifierAdapter":
        """Create a classifier adapter from a serialized artifact.

        Args:
            model_path: Path to the Joblib artifact.

        Returns:
            Initialized SklearnClassifierAdapter.

        Raises:
            FileNotFoundError: If the model file does not exist.
            KeyError: If required keys are missing from the artifact.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        artifact = joblib.load(model_path)

        model = artifact["model"]

        labels = {
            int(label_id): label_name
            for label_id, label_name in artifact["labels"].items()
        }

        model_version = artifact.get(
            "model_version",
            "unknown",
        )

        return cls(
            model=model,
            labels=labels,
            model_version=model_version,
        )

    def predict(self, text: str) -> ClassificationResult:
        """Predict the medical category for a given text.

        Args:
            text: Medical abstract or report to classify.

        Returns:
            ClassificationResult with predicted class information.
        """
        predicted_label = int(self._model.predict([text])[0])

        confidence = self._get_confidence(text)

        return ClassificationResult(
            label_id=predicted_label,
            label_name=self._labels[predicted_label],
            confidence=confidence,
            model_version=self._model_version,
        )

    def _get_confidence(self, text: str) -> float | None:
        """Calculate model confidence when probability output is available.

        Args:
            text: Medical text used for prediction.

        Returns:
            Highest predicted probability or None when the model does
            not implement ``predict_proba``.
        """
        if not hasattr(self._model, "predict_proba"):
            return None

        probabilities = self._model.predict_proba([text])[0]

        return float(max(probabilities))
