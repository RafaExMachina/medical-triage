"""Measure end-to-end latency of the medical classification API."""

import argparse
import json
import statistics
from time import perf_counter

import httpx

DEFAULT_URL = "http://localhost:8000/predict"

SAMPLE_TEXT = (
    "The patient presented with acute chest pain, "
    "myocardial infarction and coronary artery disease."
)


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Calculate an approximate percentile from latency measurements.

    Args:
        values: Collection of latency values in milliseconds.
        percentile_value: Desired percentile between 0 and 100.

    Returns:
        Latency corresponding to the requested percentile.

    Raises:
        ValueError: If the values collection is empty.
    """
    if not values:
        raise ValueError("Cannot calculate percentile from an empty collection.")

    ordered = sorted(values)

    index = round((percentile_value / 100) * (len(ordered) - 1))

    return ordered[index]


def execute_warmup(
    client: httpx.Client,
    url: str,
    requests: int,
) -> None:
    """Execute warm-up requests without recording latency.

    Args:
        client: HTTP client used to call the API.
        url: Prediction endpoint URL.
        requests: Number of warm-up requests.
    """
    payload = {
        "text": SAMPLE_TEXT,
    }

    for _ in range(requests):
        response = client.post(
            url,
            json=payload,
        )

        response.raise_for_status()


def collect_latency(
    client: httpx.Client,
    url: str,
    requests: int,
) -> list[float]:
    """Collect end-to-end API latency measurements.

    Args:
        client: HTTP client used to call the API.
        url: Prediction endpoint URL.
        requests: Number of requests to measure.

    Returns:
        List containing request latency values in milliseconds.
    """
    payload = {
        "text": SAMPLE_TEXT,
    }

    durations: list[float] = []

    for _ in range(requests):
        start_time = perf_counter()

        response = client.post(
            url,
            json=payload,
        )

        response.raise_for_status()

        elapsed_ms = (perf_counter() - start_time) * 1000

        durations.append(elapsed_ms)

    return durations


def calculate_statistics(
    durations: list[float],
) -> dict[str, float | int]:
    """Calculate latency statistics from measurements.

    Args:
        durations: Collection of request latency values.

    Returns:
        Dictionary containing mean, median, minimum, maximum,
        p95, and p99 latency metrics.
    """
    return {
        "requests": len(durations),
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "p95_ms": percentile(
            durations,
            95,
        ),
        "p99_ms": percentile(
            durations,
            99,
        ),
    }


def measure(
    url: str,
    runs: int,
    warmup: int,
) -> None:
    """Execute the complete latency benchmark.

    Args:
        url: Prediction endpoint URL.
        runs: Number of measured requests.
        warmup: Number of warm-up requests.
    """
    with httpx.Client(timeout=10.0) as client:
        print(f"Warm-up: {warmup} requests")

        execute_warmup(
            client=client,
            url=url,
            requests=warmup,
        )

        print(f"Benchmark: {runs} requests")

        durations = collect_latency(
            client=client,
            url=url,
            requests=runs,
        )

    results = calculate_statistics(durations)

    print()
    print(
        json.dumps(
            results,
            indent=2,
        )
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace containing benchmark configuration.
    """
    parser = argparse.ArgumentParser(
        description=("Measure end-to-end latency of the medical classification API.")
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Prediction endpoint URL.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of measured requests.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up requests.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the latency benchmark from command-line arguments."""
    args = parse_arguments()

    measure(
        url=args.url,
        runs=args.runs,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
