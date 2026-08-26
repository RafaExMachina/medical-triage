"""Apache Airflow DAG for the medical triage training pipeline."""

import json

import pendulum
from airflow.sdk import dag, task

from medical_triage.training.train import (
    METRICS_PATH,
    MODEL_PATH,
    load_labels,
    load_training_data,
    prepare_dataset,
    run_training,
)

DAG_ID = "medical_triage_training"


@dag(
    dag_id=DAG_ID,
    description=(
        "Prepare data, train, and validate the medical text classification baseline."
    ),
    schedule=None,
    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="UTC",
    ),
    catchup=False,
    max_active_runs=1,
    tags=[
        "medical-triage",
        "training",
        "mlops",
    ],
)
def medical_triage_training() -> None:
    """Define the medical triage model training workflow."""

    @task(
        retries=2,
        retry_delay=pendulum.duration(minutes=1),
    )
    def prepare_dataset_task() -> None:
        """Ensure that all required dataset files are available."""
        prepare_dataset()

    @task
    def validate_dataset_task() -> dict[str, int]:
        """Validate the dataset before model training.

        Returns:
            Dictionary containing the number of samples and classes.

        Raises:
            ValueError: If the dataset is empty or contains
                inconsistent class labels.
        """
        dataframe = load_training_data()
        labels = load_labels()

        sample_count = len(dataframe)
        class_count = dataframe["condition_label"].nunique()

        if sample_count == 0:
            raise ValueError("Training dataset is empty.")

        if class_count != len(labels):
            raise ValueError("Dataset classes do not match the label mapping.")

        dataset_labels = {int(label) for label in dataframe["condition_label"].unique()}

        expected_labels = set(labels)

        unknown_labels = dataset_labels.difference(expected_labels)

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found in training dataset: {unknown_labels}"
            )

        return {
            "samples": sample_count,
            "classes": class_count,
        }

    @task
    def train_model_task() -> dict[str, float]:
        """Train the model and persist its artifacts.

        Returns:
            Validation metrics generated during model training.
        """
        return run_training()

    @task
    def validate_artifacts_task(
        metrics: dict[str, float],
    ) -> None:
        """Validate artifacts produced by model training.

        Args:
            metrics: Validation metrics returned by the training task.

        Raises:
            FileNotFoundError: If an expected artifact does not exist.
            ValueError: If an artifact or metric is invalid.
        """
        required_artifacts = [
            MODEL_PATH,
            METRICS_PATH,
        ]

        missing_artifacts = [path for path in required_artifacts if not path.exists()]

        if missing_artifacts:
            raise FileNotFoundError(f"Missing training artifacts: {missing_artifacts}")

        empty_artifacts = [
            path for path in required_artifacts if path.stat().st_size == 0
        ]

        if empty_artifacts:
            raise ValueError(f"Empty training artifacts: {empty_artifacts}")

        required_metrics = {
            "accuracy",
            "f1_macro",
        }

        missing_metrics = required_metrics.difference(metrics)

        if missing_metrics:
            raise ValueError(f"Missing validation metrics: {missing_metrics}")

        for metric_name in required_metrics:
            metric_value = metrics[metric_name]

            if not 0.0 <= metric_value <= 1.0:
                raise ValueError(f"Invalid metric value: {metric_name}={metric_value}")

        persisted_metrics = json.loads(
            METRICS_PATH.read_text(
                encoding="utf-8",
            )
        )

        if persisted_metrics != metrics:
            raise ValueError("Persisted metrics differ from training metrics.")

    dataset_ready = prepare_dataset_task()
    dataset_valid = validate_dataset_task()
    training_metrics = train_model_task()
    artifacts_valid = validate_artifacts_task(training_metrics)

    dataset_ready >> dataset_valid >> training_metrics
    training_metrics >> artifacts_valid


medical_triage_training()
