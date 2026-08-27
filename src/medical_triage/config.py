"""Application configuration and environment-variable management."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Store application configuration values.

    Attributes:
        model_path: Path to the serialized machine-learning model.
        app_name: Human-readable application name.
        app_version: Current API version.
    """

    model_path: Path
    app_name: str
    app_version: str


def get_settings() -> Settings:
    """Load application settings from environment variables.

    Environment variables are optional. Default values are provided
    for local development.

    Returns:
        Settings instance containing application configuration.
    """
    return Settings(
        model_path=Path(
            os.getenv(
                "MODEL_PATH",
                "models/classifier.joblib",
            )
        ),
        app_name=os.getenv(
            "APP_NAME",
            "Medical Triage API",
        ),
        app_version=os.getenv(
            "APP_VERSION",
            "0.3.0",
        ),
    )
