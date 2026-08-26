"""Unit tests for the reusable training workflow."""

from typing import Any

import pandas as pd
import pytest

from medical_triage.training import train


def test_prepare_dataset_should_delegate_to_dataset_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that dataset preparation delegates to DatasetLoader."""
    prepare_called = False

    class FakeDatasetLoader:
        """Provide a fake dataset loader for the unit test."""

        def prepare(self) -> None:
            """Record the dataset preparation call."""
            nonlocal prepare_called
            prepare_called = True

    monkeypatch.setattr(
        train,
        "DatasetLoader",
        FakeDatasetLoader,
    )

    train.prepare_dataset()

    assert prepare_called is True


def test_run_training_should_return_metrics_and_save_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestration of training, evaluation, and persistence."""
    dataframe = pd.DataFrame(
        {
            "medical_abstract": [
                "Medical text example one.",
                "Medical text example two.",
                "Medical text example three.",
                "Medical text example four.",
            ],
            "condition_label": [
                1,
                1,
                2,
                2,
            ],
        }
    )

    labels = {
        1: "category one",
        2: "category two",
    }

    expected_metrics = {
        "accuracy": 0.75,
        "f1_macro": 0.70,
    }

    saved_model: list[Any] = []
    saved_metrics: list[dict[str, float]] = []

    class FakeModel:
        """Provide a fake trainable classification model."""

        def fit(
            self,
            x_train: pd.Series,
            y_train: pd.Series,
        ) -> None:
            """Simulate model training."""

    fake_model = FakeModel()

    monkeypatch.setattr(
        train,
        "load_training_data",
        lambda: dataframe,
    )

    monkeypatch.setattr(
        train,
        "load_labels",
        lambda: labels,
    )

    monkeypatch.setattr(
        train,
        "train_test_split",
        lambda *args, **kwargs: (
            dataframe["medical_abstract"].iloc[:2],
            dataframe["medical_abstract"].iloc[2:],
            dataframe["condition_label"].iloc[:2],
            dataframe["condition_label"].iloc[2:],
        ),
    )

    monkeypatch.setattr(
        train,
        "build_model",
        lambda: fake_model,
    )

    monkeypatch.setattr(
        train,
        "evaluate_model",
        lambda **kwargs: expected_metrics,
    )

    monkeypatch.setattr(
        train,
        "save_model",
        lambda model, labels: saved_model.append((model, labels)),
    )

    monkeypatch.setattr(
        train,
        "save_metrics",
        lambda metrics: saved_metrics.append(metrics),
    )

    result = train.run_training()

    assert result == expected_metrics

    assert saved_model == [
        (
            fake_model,
            labels,
        )
    ]

    assert saved_metrics == [
        expected_metrics,
    ]
