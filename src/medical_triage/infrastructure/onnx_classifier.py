"""ONNX Runtime adapter for medical text classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from medical_triage.domain.entities import ClassificationResult

ONNX_INTRA_OP_THREADS = 1
ONNX_INTER_OP_THREADS = 1

LABEL_OUTPUT_NAME = "label"
PROBABILITIES_OUTPUT_NAME = "probabilities"


class OnnxClassifierAdapter:
    """Adapt an ONNX model to the ClassifierPort contract.

    The complete text-classification pipeline is executed by ONNX Runtime,
    including TF-IDF preprocessing and LogisticRegression inference.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        labels: dict[int, str],
        model_version: str,
    ) -> None:
        """Initialize the ONNX classifier adapter.

        Args:
            session: Initialized ONNX Runtime inference session.
            labels: Mapping between numeric labels and readable names.
            model_version: Identifier of the loaded model.

        Raises:
            ValueError: If the ONNX graph does not expose the expected
                input and output tensors.
        """
        inputs = session.get_inputs()

        if len(inputs) != 1:
            msg = (
                "Expected ONNX model to expose exactly one input, "
                f"but found {len(inputs)}."
            )
            raise ValueError(msg)

        output_names = {output.name for output in session.get_outputs()}

        required_outputs = {
            LABEL_OUTPUT_NAME,
            PROBABILITIES_OUTPUT_NAME,
        }

        missing_outputs = required_outputs - output_names

        if missing_outputs:
            msg = f"ONNX model is missing required outputs: {sorted(missing_outputs)}"
            raise ValueError(msg)

        self._session = session
        self._labels = labels
        self._model_version = model_version
        self._input_name = inputs[0].name

    @classmethod
    def from_files(
        cls,
        model_path: Path,
        metadata_path: Path,
    ) -> OnnxClassifierAdapter:
        """Create an ONNX classifier from model and metadata files.

        Args:
            model_path: Path to the ONNX model artifact.
            metadata_path: Path to the JSON metadata artifact.

        Returns:
            Initialized OnnxClassifierAdapter.

        Raises:
            FileNotFoundError: If model or metadata files do not exist.
            TypeError: If metadata has an unexpected structure.
            KeyError: If required metadata fields are missing.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"ONNX metadata file not found: {metadata_path}")

        metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(
                encoding="utf-8",
            )
        )

        raw_labels = metadata["labels"]

        if not isinstance(raw_labels, dict):
            msg = "Expected ONNX metadata 'labels' to contain a dictionary."
            raise TypeError(msg)

        labels = {
            int(label_id): str(label_name)
            for label_id, label_name in raw_labels.items()
        }

        model_version = str(metadata["model_version"])

        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        session_options.inter_op_num_threads = ONNX_INTER_OP_THREADS
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        return cls(
            session=session,
            labels=labels,
            model_version=model_version,
        )

    def predict(
        self,
        text: str,
    ) -> ClassificationResult:
        """Predict the medical category for a given text.

        Args:
            text: Medical abstract or report to classify.

        Returns:
            ClassificationResult containing the predicted category,
            confidence, and model version.
        """
        onnx_input = np.asarray(
            [[text]],
            dtype=object,
        )

        (
            predicted_labels,
            probabilities,
        ) = self._session.run(
            [
                LABEL_OUTPUT_NAME,
                PROBABILITIES_OUTPUT_NAME,
            ],
            {
                self._input_name: onnx_input,
            },
        )

        predicted_label = int(
            np.asarray(
                predicted_labels,
            ).reshape(-1)[0]
        )

        probability_array = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        confidence = float(np.max(probability_array[0]))

        return ClassificationResult(
            label_id=predicted_label,
            label_name=self._labels[predicted_label],
            confidence=confidence,
            model_version=self._model_version,
        )
