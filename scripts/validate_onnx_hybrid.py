"""Validate sklearn TF-IDF with an ONNX LogisticRegression classifier."""

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
ONNX_HEAD_PATH = MODEL_DIR / "classifier_head.onnx"
TEST_DATASET_PATH = DATA_DIR / "medical_tc_test.csv"
REPORT_PATH = REPORT_DIR / "onnx_hybrid_equivalence.json"

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"

BATCH_SIZE = 128


def load_artifact() -> dict[str, Any]:
    """Load and validate the persisted sklearn artifact."""
    artifact = joblib.load(JOBLIB_PATH)

    if not isinstance(artifact, dict):
        msg = "Expected classifier.joblib to contain a dictionary."
        raise TypeError(msg)

    required_keys = {"model", "labels", "model_version"}
    missing_keys = required_keys - artifact.keys()

    if missing_keys:
        msg = f"Missing artifact keys: {sorted(missing_keys)}"
        raise ValueError(msg)

    return artifact


def load_test_dataset() -> pd.DataFrame:
    """Load and validate the official test dataset."""
    dataframe = pd.read_csv(TEST_DATASET_PATH)

    required_columns = {
        TEXT_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        msg = f"Test dataset is missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    if dataframe[list(required_columns)].isna().any().any():
        msg = "Test dataset contains null values."
        raise ValueError(msg)

    return dataframe


def create_onnx_session() -> ort.InferenceSession:
    """Create the ONNX Runtime session for the classifier head."""
    return ort.InferenceSession(
        str(ONNX_HEAD_PATH),
        providers=["CPUExecutionProvider"],
    )


def run_hybrid_inference(
    texts: list[str],
    tfidf: Any,
    session: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sklearn TF-IDF followed by ONNX classifier inference."""
    input_name = session.get_inputs()[0].name

    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]

        sparse_features = tfidf.transform(batch_texts)

        dense_features = sparse_features.astype(np.float32).toarray()

        batch_predictions, batch_probabilities = session.run(
            None,
            {
                input_name: dense_features,
            },
        )

        predictions.append(
            np.asarray(
                batch_predictions,
                dtype=np.int64,
            )
        )

        probabilities.append(
            np.asarray(
                batch_probabilities,
                dtype=np.float64,
            )
        )

    return (
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def run_validation() -> dict[str, Any]:
    """Compare sklearn classifier and ONNX classifier on identical features."""
    dataframe = load_test_dataset()

    texts = dataframe[TEXT_COLUMN].astype(str).tolist()

    targets = dataframe[TARGET_COLUMN].to_numpy(
        dtype=np.int64,
    )

    artifact = load_artifact()
    pipeline = artifact["model"]

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    print("Generating TF-IDF features with sklearn...")
    sklearn_features = tfidf.transform(texts)

    print("Running sklearn LogisticRegression...")
    sklearn_predictions = np.asarray(
        classifier.predict(sklearn_features),
        dtype=np.int64,
    )

    sklearn_probabilities = np.asarray(
        classifier.predict_proba(sklearn_features),
        dtype=np.float64,
    )

    print("Running hybrid sklearn TF-IDF + ONNX LogisticRegression...")
    session = create_onnx_session()

    hybrid_predictions, hybrid_probabilities = run_hybrid_inference(
        texts=texts,
        tfidf=tfidf,
        session=session,
    )

    if sklearn_predictions.shape != hybrid_predictions.shape:
        msg = (
            "Prediction shapes differ: "
            f"sklearn={sklearn_predictions.shape}, "
            f"hybrid={hybrid_predictions.shape}"
        )
        raise ValueError(msg)

    if sklearn_probabilities.shape != hybrid_probabilities.shape:
        msg = (
            "Probability shapes differ: "
            f"sklearn={sklearn_probabilities.shape}, "
            f"hybrid={hybrid_probabilities.shape}"
        )
        raise ValueError(msg)

    same_predictions = sklearn_predictions == hybrid_predictions

    matching_predictions = int(np.sum(same_predictions))

    different_predictions = int(np.sum(~same_predictions))

    agreement = float(np.mean(same_predictions))

    probability_difference = np.abs(sklearn_probabilities - hybrid_probabilities)

    maximum_probability_difference = float(np.max(probability_difference))

    mean_probability_difference = float(np.mean(probability_difference))

    probabilities_close = bool(
        np.allclose(
            sklearn_probabilities,
            hybrid_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    sklearn_accuracy = float(
        accuracy_score(
            targets,
            sklearn_predictions,
        )
    )

    hybrid_accuracy = float(
        accuracy_score(
            targets,
            hybrid_predictions,
        )
    )

    sklearn_macro_f1 = float(
        f1_score(
            targets,
            sklearn_predictions,
            average="macro",
        )
    )

    hybrid_macro_f1 = float(
        f1_score(
            targets,
            hybrid_predictions,
            average="macro",
        )
    )

    report = {
        "dataset": {
            "path": str(TEST_DATASET_PATH),
            "samples": len(dataframe),
        },
        "architecture": {
            "preprocessing": "sklearn-tfidf",
            "baseline_classifier": ("sklearn-logistic-regression"),
            "optimized_classifier": ("onnx-logistic-regression"),
            "batch_size": BATCH_SIZE,
        },
        "prediction_equivalence": {
            "matching_predictions": (matching_predictions),
            "different_predictions": (different_predictions),
            "agreement": agreement,
        },
        "quality": {
            "sklearn": {
                "accuracy": sklearn_accuracy,
                "macro_f1": sklearn_macro_f1,
            },
            "hybrid": {
                "accuracy": hybrid_accuracy,
                "macro_f1": hybrid_macro_f1,
            },
        },
        "probability_equivalence": {
            "allclose": probabilities_close,
            "rtol": 1e-5,
            "atol": 1e-6,
            "maximum_absolute_difference": (maximum_probability_difference),
            "mean_absolute_difference": (mean_probability_difference),
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    """Print hybrid equivalence results."""
    dataset = report["dataset"]
    equivalence = report["prediction_equivalence"]
    quality = report["quality"]
    probabilities = report["probability_equivalence"]

    print()
    print("=== DATASET ===")
    print(f"Samples: {dataset['samples']}")

    print()
    print("=== HYBRID PREDICTION EQUIVALENCE ===")
    print(f"Matching predictions: {equivalence['matching_predictions']}")
    print(f"Different predictions: {equivalence['different_predictions']}")
    print(f"Agreement: {equivalence['agreement'] * 100:.6f}%")

    print()
    print("=== QUALITY ===")
    print(f"Sklearn accuracy: {quality['sklearn']['accuracy']:.6f}")
    print(f"Hybrid accuracy:  {quality['hybrid']['accuracy']:.6f}")
    print(f"Sklearn macro F1: {quality['sklearn']['macro_f1']:.6f}")
    print(f"Hybrid macro F1:  {quality['hybrid']['macro_f1']:.6f}")

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
    """Run the hybrid ONNX equivalence validation."""
    report = run_validation()
    print_report(report)


if __name__ == "__main__":
    main()
