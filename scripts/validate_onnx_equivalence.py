"""Validate prediction equivalence between sklearn and ONNX models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

JOBLIB_PATH = MODEL_DIR / "classifier.joblib"
ONNX_PATH = MODEL_DIR / "classifier.onnx"
TEST_DATASET_PATH = DATA_DIR / "medical_tc_test.csv"
REPORT_PATH = REPORT_DIR / "onnx_equivalence.json"

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"


def load_sklearn_model() -> Any:
    """Load the persisted sklearn pipeline."""
    artifact = joblib.load(JOBLIB_PATH)

    if not isinstance(artifact, dict):
        msg = "Expected classifier.joblib to contain a dictionary."
        raise TypeError(msg)

    if "model" not in artifact:
        msg = "The persisted artifact does not contain the 'model' key."
        raise ValueError(msg)

    return artifact["model"]


def load_test_dataset() -> pd.DataFrame:
    """Load and validate the official test dataset."""
    dataframe = pd.read_csv(TEST_DATASET_PATH)

    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        msg = f"Test dataset is missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    if dataframe[[TEXT_COLUMN, TARGET_COLUMN]].isna().any().any():
        msg = "Test dataset contains null values."
        raise ValueError(msg)

    return dataframe


def create_onnx_session() -> ort.InferenceSession:
    """Create an ONNX Runtime CPU inference session."""
    return ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )


def run_validation() -> dict[str, Any]:
    """Compare sklearn and ONNX predictions on the complete test dataset."""
    dataframe = load_test_dataset()

    texts = dataframe[TEXT_COLUMN].astype(str).tolist()
    targets = dataframe[TARGET_COLUMN].to_numpy(dtype=np.int64)

    sklearn_model = load_sklearn_model()

    print("Running sklearn inference...")
    sklearn_predictions = np.asarray(
        sklearn_model.predict(texts),
        dtype=np.int64,
    )
    sklearn_probabilities = np.asarray(
        sklearn_model.predict_proba(texts),
        dtype=np.float64,
    )

    print("Running ONNX Runtime inference...")
    session = create_onnx_session()

    input_name = session.get_inputs()[0].name
    onnx_input = np.asarray(texts, dtype=object).reshape(-1, 1)

    onnx_predictions_raw, onnx_probabilities_raw = session.run(
        None,
        {input_name: onnx_input},
    )

    onnx_predictions = np.asarray(
        onnx_predictions_raw,
        dtype=np.int64,
    )
    onnx_probabilities = np.asarray(
        onnx_probabilities_raw,
        dtype=np.float64,
    )

    if sklearn_predictions.shape != onnx_predictions.shape:
        msg = (
            "Prediction shapes differ: "
            f"sklearn={sklearn_predictions.shape}, "
            f"onnx={onnx_predictions.shape}"
        )
        raise ValueError(msg)

    if sklearn_probabilities.shape != onnx_probabilities.shape:
        msg = (
            "Probability shapes differ: "
            f"sklearn={sklearn_probabilities.shape}, "
            f"onnx={onnx_probabilities.shape}"
        )
        raise ValueError(msg)

    same_predictions = sklearn_predictions == onnx_predictions

    matching_predictions = int(np.sum(same_predictions))
    different_predictions = int(np.sum(~same_predictions))
    agreement = float(np.mean(same_predictions))

    probability_difference = np.abs(sklearn_probabilities - onnx_probabilities)

    max_probability_difference = float(np.max(probability_difference))
    mean_probability_difference = float(np.mean(probability_difference))

    probabilities_close = bool(
        np.allclose(
            sklearn_probabilities,
            onnx_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    sklearn_accuracy = float(accuracy_score(targets, sklearn_predictions))
    onnx_accuracy = float(accuracy_score(targets, onnx_predictions))

    sklearn_macro_f1 = float(
        f1_score(
            targets,
            sklearn_predictions,
            average="macro",
        )
    )
    onnx_macro_f1 = float(
        f1_score(
            targets,
            onnx_predictions,
            average="macro",
        )
    )

    report = {
        "dataset": {
            "path": str(TEST_DATASET_PATH),
            "samples": len(dataframe),
            "text_column": TEXT_COLUMN,
            "target_column": TARGET_COLUMN,
        },
        "prediction_equivalence": {
            "matching_predictions": matching_predictions,
            "different_predictions": different_predictions,
            "agreement": agreement,
        },
        "quality": {
            "sklearn": {
                "accuracy": sklearn_accuracy,
                "macro_f1": sklearn_macro_f1,
            },
            "onnx": {
                "accuracy": onnx_accuracy,
                "macro_f1": onnx_macro_f1,
            },
        },
        "probability_equivalence": {
            "allclose": probabilities_close,
            "rtol": 1e-5,
            "atol": 1e-6,
            "maximum_absolute_difference": max_probability_difference,
            "mean_absolute_difference": mean_probability_difference,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print the equivalence report in a human-readable format."""
    dataset = report["dataset"]
    equivalence = report["prediction_equivalence"]
    quality = report["quality"]
    probabilities = report["probability_equivalence"]

    print()
    print("=== DATASET ===")
    print(f"Samples: {dataset['samples']}")

    print()
    print("=== PREDICTION EQUIVALENCE ===")
    print(f"Matching predictions: {equivalence['matching_predictions']}")
    print(f"Different predictions: {equivalence['different_predictions']}")
    print(f"Agreement: {equivalence['agreement'] * 100:.6f}%")

    print()
    print("=== QUALITY ===")
    print(f"Sklearn accuracy: {quality['sklearn']['accuracy']:.6f}")
    print(f"ONNX accuracy:    {quality['onnx']['accuracy']:.6f}")
    print(f"Sklearn macro F1: {quality['sklearn']['macro_f1']:.6f}")
    print(f"ONNX macro F1:    {quality['onnx']['macro_f1']:.6f}")

    print()
    print("=== PROBABILITY EQUIVALENCE ===")
    print(f"Probabilities close: {probabilities['allclose']}")
    print(
        "Maximum absolute difference: "
        f"{probabilities['maximum_absolute_difference']:.12e}"
    )
    print(
        f"Mean absolute difference:    {probabilities['mean_absolute_difference']:.12e}"
    )

    print()
    print(f"Report: {REPORT_PATH}")


def main() -> None:
    """Run complete sklearn versus ONNX validation."""
    report = run_validation()
    print_report(report)


if __name__ == "__main__":
    main()
