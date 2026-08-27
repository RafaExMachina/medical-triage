"""Consolidate HTTP benchmark reports for sklearn and ONNX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

REPORT_DIR = Path("reports")
OUTPUT_PATH = REPORT_DIR / "http_benchmark_summary.json"

BACKENDS = ("sklearn", "onnx")

SECTIONS = (
    "client_latency_ms",
    "server_inference_ms",
    "application_overhead_ms",
)

STATISTICS = (
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
)


def load_report(path: Path) -> dict[str, Any]:
    """Load one benchmark JSON report."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark report not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark report: {path}")

    return cast(dict[str, Any], data)


def average_section(
    reports: list[dict[str, Any]],
    section: str,
) -> dict[str, float]:
    """Average statistics from the same section across runs."""
    result: dict[str, float] = {}

    for statistic in STATISTICS:
        values = [float(report[section][statistic]) for report in reports]

        result[statistic] = sum(values) / len(values)

    return result


def consolidate_backend(
    backend: str,
) -> dict[str, Any]:
    """Consolidate the three benchmark runs for one backend."""
    paths = [REPORT_DIR / f"http_benchmark_{backend}_{run}.json" for run in range(1, 4)]

    reports = [load_report(path) for path in paths]

    return {
        "runs": len(reports),
        "source_reports": [str(path) for path in paths],
        **{
            section: average_section(
                reports,
                section,
            )
            for section in SECTIONS
        },
    }


def compare_metrics(
    sklearn: dict[str, float],
    onnx: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compare ONNX metrics against the sklearn baseline."""
    comparison: dict[str, dict[str, float]] = {}

    for statistic in STATISTICS:
        sklearn_value = sklearn[statistic]
        onnx_value = onnx[statistic]

        comparison[statistic] = {
            "sklearn_ms": sklearn_value,
            "onnx_ms": onnx_value,
            "speedup_x": (sklearn_value / onnx_value if onnx_value > 0 else 0.0),
            "reduction_percent": (
                (1.0 - (onnx_value / sklearn_value)) * 100.0
                if sklearn_value > 0
                else 0.0
            ),
        }

    return comparison


def main() -> None:
    """Build and persist the consolidated benchmark report."""
    sklearn = consolidate_backend("sklearn")
    onnx = consolidate_backend("onnx")

    summary = {
        "methodology": {
            "backends": list(BACKENDS),
            "runs_per_backend": 3,
            "requests_per_run": 1000,
            "warmup_requests_per_run": 50,
            "request_mode": "single-text",
            "http_connection": "persistent",
            "aggregation": ("arithmetic mean of per-run summary statistics"),
            "percentile_note": (
                "p50, p95 and p99 are averages "
                "of per-run percentiles, not "
                "percentiles pooled from raw samples"
            ),
        },
        "sklearn": sklearn,
        "onnx": onnx,
        "comparison": {
            "client_latency_ms": compare_metrics(
                sklearn["client_latency_ms"],
                onnx["client_latency_ms"],
            ),
            "server_inference_ms": compare_metrics(
                sklearn["server_inference_ms"],
                onnx["server_inference_ms"],
            ),
            "application_overhead_ms": compare_metrics(
                sklearn["application_overhead_ms"],
                onnx["application_overhead_ms"],
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(f"Summary written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
