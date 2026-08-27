"""Benchmark end-to-end HTTP and server inference latency."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/predict"
DEFAULT_DATASET = Path("data/raw/medical_tc_test.csv")

TEXT_COLUMN = "medical_abstract"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line benchmark arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HTTP and server-side inference latency "
            "for the Medical Triage API."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Prediction endpoint URL.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of measured requests.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warm-up requests.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV dataset containing medical texts.",
    )

    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "sklearn",
            "onnx",
        ],
        help="Inference backend being benchmarked.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file where benchmark results will be stored.",
    )

    return parser.parse_args()


def load_texts(
    dataset_path: Path,
) -> list[str]:
    """Load medical texts from the benchmark dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts: list[str] = []

    with dataset_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            msg = "Dataset does not contain a CSV header."
            raise ValueError(msg)

        if TEXT_COLUMN not in reader.fieldnames:
            msg = f"Dataset does not contain required column: {TEXT_COLUMN}"
            raise ValueError(msg)

        for row in reader:
            text = row.get(TEXT_COLUMN)

            if text:
                texts.append(text)

    if not texts:
        msg = "Dataset does not contain benchmark texts."
        raise ValueError(msg)

    return texts


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Calculate an approximate percentile."""
    if not values:
        msg = "Cannot calculate percentile from an empty collection."
        raise ValueError(msg)

    ordered = sorted(values)

    index = round((percentile_value / 100) * (len(ordered) - 1))

    return ordered[index]


def calculate_statistics(
    values: list[float],
) -> dict[str, float | int]:
    """Calculate latency statistics."""
    mean_ms = statistics.mean(values)

    return {
        "samples": len(values),
        "mean_ms": mean_ms,
        "p50_ms": percentile(
            values,
            50,
        ),
        "p95_ms": percentile(
            values,
            95,
        ),
        "p99_ms": percentile(
            values,
            99,
        ),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": (statistics.stdev(values) if len(values) > 1 else 0.0),
        "estimated_requests_per_second": (1000.0 / mean_ms if mean_ms > 0 else 0.0),
    }


def execute_request(
    client: httpx.Client,
    url: str,
    text: str,
) -> tuple[float, float]:
    """Execute one request and return client and server latency."""
    payload = {
        "text": text,
    }

    start_time = perf_counter()

    response = client.post(
        url,
        json=payload,
    )

    elapsed_ms = (perf_counter() - start_time) * 1000

    response.raise_for_status()

    response_payload: dict[str, Any] = response.json()

    if "inference_ms" not in response_payload:
        msg = "API response does not contain 'inference_ms'."
        raise ValueError(msg)

    server_inference_ms = float(response_payload["inference_ms"])

    return (
        elapsed_ms,
        server_inference_ms,
    )


def execute_warmup(
    client: httpx.Client,
    url: str,
    texts: list[str],
    requests: int,
) -> None:
    """Execute warm-up requests without recording results."""
    for index in range(requests):
        text = texts[index % len(texts)]

        execute_request(
            client=client,
            url=url,
            text=text,
        )


def collect_measurements(
    client: httpx.Client,
    url: str,
    texts: list[str],
    requests: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
]:
    """Collect HTTP, inference, and overhead measurements."""
    client_latencies: list[float] = []
    inference_latencies: list[float] = []
    overhead_latencies: list[float] = []

    for index in range(requests):
        text = texts[index % len(texts)]

        (
            client_ms,
            inference_ms,
        ) = execute_request(
            client=client,
            url=url,
            text=text,
        )

        client_latencies.append(
            client_ms,
        )

        inference_latencies.append(
            inference_ms,
        )

        overhead_latencies.append(
            client_ms - inference_ms,
        )

    return (
        client_latencies,
        inference_latencies,
        overhead_latencies,
    )


def run_benchmark(
    *,
    url: str,
    dataset_path: Path,
    backend: str,
    runs: int,
    warmup: int,
    output_path: Path,
) -> dict[str, Any]:
    """Execute the complete HTTP benchmark."""
    texts = load_texts(
        dataset_path,
    )

    print("Benchmark configuration:")
    print(f"  backend: {backend}")
    print(f"  runs: {runs}")
    print(f"  warmup: {warmup}")
    print(f"  dataset samples: {len(texts)}")
    print(f"  url: {url}")

    with httpx.Client(
        timeout=30.0,
    ) as client:
        print()
        print(f"Warm-up: {warmup} requests")

        execute_warmup(
            client=client,
            url=url,
            texts=texts,
            requests=warmup,
        )

        print(f"Benchmark: {runs} requests")

        (
            client_latencies,
            inference_latencies,
            overhead_latencies,
        ) = collect_measurements(
            client=client,
            url=url,
            texts=texts,
            requests=runs,
        )

    report: dict[str, Any] = {
        "configuration": {
            "backend": backend,
            "url": url,
            "runs": runs,
            "warmup": warmup,
            "dataset": str(dataset_path),
            "dataset_samples": len(texts),
            "request_mode": "single-text",
            "http_connection": "persistent",
        },
        "client_latency_ms": (calculate_statistics(client_latencies)),
        "server_inference_ms": (calculate_statistics(inference_latencies)),
        "application_overhead_ms": (calculate_statistics(overhead_latencies)),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def print_report(
    report: dict[str, Any],
) -> None:
    """Print benchmark results."""
    backend = report["configuration"]["backend"]

    print()
    print("=== HTTP API BENCHMARK ===")
    print(f"Backend: {backend}")

    sections = {
        "CLIENT HTTP": (report["client_latency_ms"]),
        "SERVER INFERENCE": (report["server_inference_ms"]),
        "APPLICATION OVERHEAD": (report["application_overhead_ms"]),
    }

    for name, result in sections.items():
        print()
        print(name)

        print(f"  mean: {result['mean_ms']:.4f} ms")
        print(f"  p50:  {result['p50_ms']:.4f} ms")
        print(f"  p95:  {result['p95_ms']:.4f} ms")
        print(f"  p99:  {result['p99_ms']:.4f} ms")
        print(f"  min:  {result['min_ms']:.4f} ms")
        print(f"  max:  {result['max_ms']:.4f} ms")
        print(f"  stdev: {result['stdev_ms']:.4f} ms")

    print()
    print(
        "Estimated HTTP throughput: "
        f"{report['client_latency_ms']['estimated_requests_per_second']:.2f} req/s"
    )


def main() -> None:
    """Run the benchmark from command-line arguments."""
    args = parse_arguments()

    report = run_benchmark(
        url=args.url,
        dataset_path=args.dataset,
        backend=args.backend,
        runs=args.runs,
        warmup=args.warmup,
        output_path=args.output,
    )

    print_report(
        report,
    )

    print()
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
