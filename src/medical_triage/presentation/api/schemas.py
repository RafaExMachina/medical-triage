"""Pydantic schemas used by the REST API."""

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Represent an incoming medical text classification request."""

    text: str = Field(
        min_length=20,
        description="Medical abstract or report to classify.",
        examples=[
            ("The patient presented with acute chest pain and myocardial infarction.")
        ],
    )


class ClassificationResponse(BaseModel):
    """Represent the API response for a classification request."""

    label_id: int = Field(description="Numeric identifier of the predicted category.")

    label_name: str = Field(description="Human-readable predicted category.")

    confidence: float | None = Field(
        description="Prediction confidence when available."
    )

    model_version: str = Field(description="Version of the model used for inference.")

    inference_ms: float = Field(
        description="Internal inference latency in milliseconds."
    )


class HealthResponse(BaseModel):
    """Represent the API health-check response."""

    status: str
