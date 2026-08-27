"""Unit tests for application configuration."""

import pytest

from medical_triage.config import get_settings


def test_default_backend_should_be_onnx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that ONNX is the default inference backend."""
    monkeypatch.delenv(
        "MODEL_BACKEND",
        raising=False,
    )

    settings = get_settings()

    assert settings.model_backend == "onnx"


def test_sklearn_backend_should_be_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that sklearn can be selected explicitly."""
    monkeypatch.setenv(
        "MODEL_BACKEND",
        "sklearn",
    )

    settings = get_settings()

    assert settings.model_backend == "sklearn"


def test_invalid_backend_should_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that unsupported inference backends are rejected."""
    monkeypatch.setenv(
        "MODEL_BACKEND",
        "pytorch",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported MODEL_BACKEND",
    ):
        get_settings()
