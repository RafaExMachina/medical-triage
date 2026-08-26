"""Integration tests for the FastAPI HTTP interface."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from medical_triage.application.use_cases import ClassifyMedicalTextUseCase
from medical_triage.domain.entities import ClassificationResult
from medical_triage.presentation.api import main as api_main
from medical_triage.presentation.api.dependencies import (
    get_classification_use_case,
)


class FakeClassifier:
    """Provide a deterministic classifier for API integration tests."""

    def predict(self, text: str) -> ClassificationResult:
        """Return a deterministic classification result.

        Args:
            text: Medical text received by the fake classifier.

        Returns:
            Deterministic classification result used by the tests.
        """
        return ClassificationResult(
            label_id=4,
            label_name="cardiovascular diseases",
            confidence=0.95,
            model_version="test-model",
        )


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Create a FastAPI test client with a fake classifier.

    Args:
        monkeypatch: Pytest fixture used to replace the model-loading
            dependency during application startup.

    Yields:
        Configured FastAPI TestClient.
    """
    use_case = ClassifyMedicalTextUseCase(
        classifier=FakeClassifier(),
    )

    monkeypatch.setattr(
        api_main,
        "get_classification_use_case",
        lambda: use_case,
    )

    api_main.app.dependency_overrides[get_classification_use_case] = lambda: use_case

    with TestClient(api_main.app) as test_client:
        yield test_client

    api_main.app.dependency_overrides.clear()


def test_health_endpoint_should_return_healthy(
    client: TestClient,
) -> None:
    """Verify that the health endpoint reports a healthy API."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_metrics_endpoint_should_expose_prometheus_metrics(
    client,
) -> None:
    """Ensure that Prometheus metrics are exposed."""
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "medical_triage_http_requests_total" in response.text
    assert "medical_triage_http_request_duration_seconds" in response.text
    assert "medical_triage_predictions_total" in response.text


def test_predict_endpoint_should_return_classification(
    client: TestClient,
) -> None:
    """Verify that valid medical text produces a classification."""
    response = client.post(
        "/predict",
        json={
            "text": (
                "The patient presented with acute myocardial "
                "infarction and severe coronary artery disease."
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["label_id"] == 4
    assert payload["label_name"] == "cardiovascular diseases"
    assert payload["confidence"] == pytest.approx(0.95)
    assert payload["model_version"] == "test-model"
    assert payload["inference_ms"] >= 0


def test_predict_endpoint_should_reject_short_text(
    client: TestClient,
) -> None:
    """Verify that medical text shorter than the limit is rejected."""
    response = client.post(
        "/predict",
        json={
            "text": "short",
        },
    )

    assert response.status_code == 422

    payload = response.json()

    assert payload["detail"][0]["type"] == "string_too_short"
