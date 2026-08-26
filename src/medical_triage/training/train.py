"""Train and persist the baseline medical text classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from medical_triage.data.dataset_loader import DatasetLoader
from medical_triage.observability.logging import configure_logging, get_logger

TRAIN_PATH = Path("data/raw/medical_tc_train.csv")
LABELS_PATH = Path("data/raw/medical_tc_labels.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
MODEL_VERSION = "tfidf-logreg-v1"

logger = get_logger(__name__)


def prepare_dataset() -> None:
    """Ensure that all required dataset files are available locally.

    Existing dataset files are preserved. Missing files are downloaded
    by the DatasetLoader.
    """
    logger.info("Preparing dataset")

    dataset_loader = DatasetLoader()
    dataset_loader.prepare()

    logger.info("Dataset preparation completed")


def load_training_data() -> pd.DataFrame:
    """Load and validate the training dataset from disk.

    Returns:
        DataFrame containing medical abstracts and their labels.

    Raises:
        FileNotFoundError: If the training CSV file does not exist.
        ValueError: If required columns are missing or the dataset
            contains no valid samples.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

    dataframe = pd.read_csv(TRAIN_PATH)

    required_columns = {
        "condition_label",
        "medical_abstract",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataframe = dataframe.dropna(
        subset=[
            "condition_label",
            "medical_abstract",
        ]
    ).copy()

    if dataframe.empty:
        raise ValueError("Training dataset contains no valid samples.")

    return dataframe


def load_labels() -> dict[int, str]:
    """Load the mapping between class IDs and category names.

    Returns:
        Dictionary mapping integer labels to readable category names.

    Raises:
        FileNotFoundError: If the labels CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    labels_df = pd.read_csv(LABELS_PATH)

    required_columns = {
        "condition_label",
        "condition_name",
    }

    missing_columns = required_columns.difference(labels_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return {
        int(label): str(name)
        for label, name in zip(
            labels_df["condition_label"],
            labels_df["condition_name"],
            strict=True,
        )
    }


def build_model() -> Pipeline:
    """Create the baseline NLP classification pipeline.

    The pipeline converts medical text into TF-IDF features and
    applies Logistic Regression for multiclass classification.

    Returns:
        Untrained Scikit-learn classification pipeline.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_validation: pd.Series,
    y_validation: pd.Series,
) -> dict[str, float]:
    """Evaluate the trained model using validation data.

    Args:
        model: Trained classification pipeline.
        x_validation: Medical abstracts reserved for validation.
        y_validation: Ground-truth validation labels.

    Returns:
        Dictionary containing accuracy and macro F1 metrics.
    """
    predictions = model.predict(x_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    f1_macro = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }


def save_model(
    model: Pipeline,
    labels: dict[int, str],
) -> None:
    """Serialize the trained model and its metadata.

    Args:
        model: Trained Scikit-learn classification pipeline.
        labels: Mapping between label IDs and readable names.
    """
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "labels": labels,
        "model_version": MODEL_VERSION,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    logger.info(
        "Model saved | path=%s | version=%s",
        MODEL_PATH,
        MODEL_VERSION,
    )


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Persist evaluation metrics as a JSON file.

    Args:
        metrics: Dictionary containing model evaluation metrics.
    """
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Metrics saved | path=%s",
        METRICS_PATH,
    )


def run_training() -> dict[str, float]:
    """Execute model training, evaluation, and artifact persistence.

    This function assumes that the required dataset files have already
    been prepared. It can therefore be reused by command-line execution,
    tests, or an external orchestrator such as Apache Airflow.

    Returns:
        Dictionary containing the validation metrics generated by the
        trained model.
    """
    dataframe = load_training_data()
    labels = load_labels()

    logger.info(
        "Training dataset loaded | samples=%d | classes=%d",
        len(dataframe),
        dataframe["condition_label"].nunique(),
    )

    x = dataframe["medical_abstract"]
    y = dataframe["condition_label"]

    (
        x_train,
        x_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        x,
        y,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Dataset split completed | training_samples=%d | validation_samples=%d",
        len(x_train),
        len(x_validation),
    )

    model = build_model()

    logger.info("Training model")

    model.fit(
        x_train,
        y_train,
    )

    logger.info("Model training completed")

    metrics = evaluate_model(
        model=model,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    logger.info(
        "Validation completed | accuracy=%.4f | f1_macro=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
    )

    save_model(
        model=model,
        labels=labels,
    )

    save_metrics(metrics)

    return metrics


def main() -> None:
    """Execute the complete standalone model training workflow."""
    configure_logging()

    logger.info(
        "Starting training workflow | model_version=%s",
        MODEL_VERSION,
    )

    prepare_dataset()

    run_training()

    logger.info("Training workflow completed successfully")


if __name__ == "__main__":
    main()
