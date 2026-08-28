"""Application configuration and environment-variable management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ModelBackend = Literal[
    "sklearn",
    "onnx",
]


@dataclass(frozen=True, slots=True)
class Settings:
    """Store application configuration values.

    Attributes:
        model_backend: Inference backend selected for the API.
        model_path: Path to the serialized Scikit-learn model.
        onnx_model_path: Path to the optimized ONNX model.
        onnx_metadata_path: Path to the ONNX metadata file.
        app_name: Human-readable application name.
        app_version: Current API version.
    """

    model_backend: ModelBackend
    model_path: Path
    onnx_model_path: Path
    onnx_metadata_path: Path
    app_name: str
    app_version: str


def get_model_backend() -> ModelBackend:
    """Read and validate the configured inference backend.

    Returns:
        Validated model backend.

    Raises:
        ValueError: If MODEL_BACKEND contains an unsupported value.
    """
    backend = (
        os.getenv(
            "MODEL_BACKEND",
            "onnx",
        )
        .strip()
        .lower()
    )

    supported_backends = {
        "sklearn",
        "onnx",
    }

    if backend not in supported_backends:
        msg = f"Unsupported MODEL_BACKEND {backend!r}. Expected one of: sklearn, onnx."
        raise ValueError(msg)

    return cast(
        ModelBackend,
        backend,
    )


def get_settings() -> Settings:
    """Load application settings from environment variables.

    Environment variables are optional. Default values are provided
    for local development.

    Returns:
        Settings instance containing application configuration.
    """
    return Settings(
        model_backend=get_model_backend(),
        model_path=Path(
            os.getenv(
                "MODEL_PATH",
                "models/classifier.joblib",
            )
        ),
        onnx_model_path=Path(
            os.getenv(
                "ONNX_MODEL_PATH",
                "models/classifier.onnx",
            )
        ),
        onnx_metadata_path=Path(
            os.getenv(
                "ONNX_METADATA_PATH",
                "models/classifier_onnx_metadata.json",
            )
        ),
        app_name=os.getenv(
            "APP_NAME",
            "Medical Triage API",
        ),
        app_version=os.getenv(
            "APP_VERSION",
            "0.4.0",
        ),
    )
