"""Unit tests for the ONNX classifier adapter."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from medical_triage.infrastructure.onnx_classifier import (
    OnnxClassifierAdapter,
)


class FakeTensorInfo:
    """Represent fake ONNX tensor metadata."""

    def __init__(self, name: str) -> None:
        """Initialize tensor metadata."""
        self.name = name


class FakeOnnxSession:
    """Provide deterministic ONNX Runtime behavior for tests."""

    def get_inputs(self) -> list[FakeTensorInfo]:
        """Return the expected model input."""
        return [
            FakeTensorInfo("text"),
        ]

    def get_outputs(self) -> list[FakeTensorInfo]:
        """Return the expected model outputs."""
        return [
            FakeTensorInfo("label"),
            FakeTensorInfo("probabilities"),
        ]

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        """Return deterministic prediction outputs."""
        assert output_names == [
            "label",
            "probabilities",
        ]

        assert "text" in input_feed
        assert input_feed["text"].shape == (1, 1)

        return [
            np.asarray(
                [4],
                dtype=np.int64,
            ),
            np.asarray(
                [
                    [
                        0.01,
                        0.02,
                        0.03,
                        0.90,
                        0.04,
                    ]
                ],
                dtype=np.float32,
            ),
        ]


class FakeInvalidOnnxSession:
    """Represent an ONNX graph missing required outputs."""

    def get_inputs(self) -> list[FakeTensorInfo]:
        """Return the expected model input."""
        return [
            FakeTensorInfo("text"),
        ]

    def get_outputs(self) -> list[FakeTensorInfo]:
        """Return an incomplete output definition."""
        return [
            FakeTensorInfo("label"),
        ]


def test_predict_should_return_classification_result() -> None:
    """Verify that ONNX inference is mapped to the domain entity."""
    session = cast(
        Any,
        FakeOnnxSession(),
    )

    classifier = OnnxClassifierAdapter(
        session=session,
        labels={
            1: "neoplasms",
            2: "digestive system diseases",
            3: "nervous system diseases",
            4: "cardiovascular diseases",
            5: "general pathological conditions",
        },
        model_version="test-onnx-v1",
    )

    result = classifier.predict(
        "The patient presented with acute myocardial infarction."
    )

    assert result.label_id == 4
    assert result.label_name == "cardiovascular diseases"
    assert result.confidence == pytest.approx(
        0.90,
    )
    assert result.model_version == "test-onnx-v1"


def test_constructor_should_reject_missing_outputs() -> None:
    """Verify that an invalid ONNX graph is rejected."""
    session = cast(
        Any,
        FakeInvalidOnnxSession(),
    )

    with pytest.raises(
        ValueError,
        match="missing required outputs",
    ):
        OnnxClassifierAdapter(
            session=session,
            labels={
                4: "cardiovascular diseases",
            },
            model_version="test",
        )
