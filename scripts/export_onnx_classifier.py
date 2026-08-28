"""Export only the LogisticRegression classifier to ONNX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_DIR = Path("models")

JOBLIB_PATH = MODEL_DIR / "classifier.joblib"
ONNX_PATH = MODEL_DIR / "classifier_head.onnx"
METADATA_PATH = MODEL_DIR / "classifier_head_onnx_metadata.json"

REQUESTED_TARGET_OPSET = 18


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and validate the persisted sklearn artifact."""
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        msg = "Expected classifier.joblib to contain a dictionary."
        raise TypeError(msg)

    required_keys = {"model", "labels", "model_version"}
    missing_keys = required_keys - artifact.keys()

    if missing_keys:
        msg = f"Missing artifact keys: {sorted(missing_keys)}"
        raise ValueError(msg)

    return artifact


def get_effective_opsets(
    onnx_model: onnx.ModelProto,
) -> dict[str, int]:
    """Return the opsets effectively used by the ONNX model."""
    return {
        opset.domain or "ai.onnx": opset.version for opset in onnx_model.opset_import
    }


def export_classifier() -> None:
    """Export only the trained LogisticRegression classifier to ONNX."""
    artifact = load_artifact(JOBLIB_PATH)

    pipeline = artifact["model"]
    labels = artifact["labels"]
    model_version = artifact["model_version"]

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    n_features = len(tfidf.vocabulary_)

    initial_types = [
        (
            "features",
            FloatTensorType([None, n_features]),
        )
    ]

    options = {
        id(classifier): {
            "zipmap": False,
        }
    }

    onnx_model = convert_sklearn(
        classifier,
        name="medical-triage-logistic-regression",
        initial_types=initial_types,
        options=options,
        target_opset=REQUESTED_TARGET_OPSET,
    )

    ONNX_PATH.write_bytes(
        onnx_model.SerializeToString(),
    )

    loaded_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(loaded_model)

    effective_opsets = get_effective_opsets(
        loaded_model,
    )

    metadata = {
        "model_version": model_version,
        "source_artifact": JOBLIB_PATH.name,
        "onnx_artifact": ONNX_PATH.name,
        "labels": labels,
        "optimization": "logistic-regression-to-onnx",
        "preprocessing": "sklearn-tfidf",
        "number_of_features": n_features,
        "requested_target_opset": REQUESTED_TARGET_OPSET,
        "effective_opsets": effective_opsets,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    onnx_size = ONNX_PATH.stat().st_size

    print("ONNX classifier export completed successfully.")
    print(f"Source:   {JOBLIB_PATH}")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print()
    print("Classifier:")
    print(f"  features: {n_features}")
    print(f"  classes: {classifier.classes_.tolist()}")

    print()
    print("ONNX opsets:")
    print(f"  requested target: {REQUESTED_TARGET_OPSET}")

    for domain, version in effective_opsets.items():
        print(f"  {domain}: {version}")

    print()
    print("Artifact size:")
    print(f"  ONNX classifier: {onnx_size / 1024 / 1024:.3f} MiB")


if __name__ == "__main__":
    export_classifier()
