"""Benchmark sklearn, hybrid ONNX, and full ONNX inference backends."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

JOBLIB_PATH = MODEL_DIR / "classifier.joblib"
FULL_ONNX_PATH = MODEL_DIR / "classifier.onnx"
ONNX_HEAD_PATH = MODEL_DIR / "classifier_head.onnx"

TEST_DATASET_PATH = DATA_DIR / "medical_tc_test.csv"
REPORT_PATH = REPORT_DIR / "inference_benchmark.json"

TEXT_COLUMN = "medical_abstract"

ONNX_INTRA_OP_THREADS = 1
ONNX_INTER_OP_THREADS = 1


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Benchmark sklearn, hybrid ONNX, and full ONNX inference.")
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Number of timed inference requests per backend.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup requests per backend.",
    )

    return parser.parse_args()


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Calculate a percentile using NumPy."""
    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            percentile_value,
        )
    )


def summarize_latencies(
    values_ms: list[float],
) -> dict[str, float]:
    """Calculate latency statistics in milliseconds."""
    mean_ms = statistics.mean(values_ms)

    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(values_ms, 50),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "stdev_ms": (statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0),
        "estimated_requests_per_second": (1000.0 / mean_ms if mean_ms > 0 else 0.0),
    }


def benchmark_backend(
    inference_function: Callable[[str], Any],
    texts: list[str],
    runs: int,
    warmup: int,
) -> dict[str, float]:
    """Benchmark one inference backend."""
    for index in range(warmup):
        inference_function(texts[index % len(texts)])

    latencies_ms: list[float] = []

    for index in range(runs):
        text = texts[index % len(texts)]

        start = time.perf_counter_ns()

        inference_function(text)

        end = time.perf_counter_ns()

        elapsed_ms = (end - start) / 1_000_000

        latencies_ms.append(elapsed_ms)

    return summarize_latencies(
        latencies_ms,
    )


def create_onnx_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create a controlled single-thread ONNX Runtime session."""
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = ONNX_INTRA_OP_THREADS
    session_options.inter_op_num_threads = ONNX_INTER_OP_THREADS

    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def load_resources() -> tuple[
    Any,
    Any,
    Any,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    """Load sklearn and ONNX inference resources."""
    artifact = joblib.load(
        JOBLIB_PATH,
    )

    if not isinstance(artifact, dict):
        msg = "Expected classifier.joblib to contain a dictionary."
        raise TypeError(msg)

    if "model" not in artifact:
        msg = "The persisted artifact does not contain the 'model' key."
        raise ValueError(msg)

    pipeline = artifact["model"]

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    full_onnx_session = create_onnx_session(
        FULL_ONNX_PATH,
    )

    head_onnx_session = create_onnx_session(
        ONNX_HEAD_PATH,
    )

    return (
        pipeline,
        tfidf,
        classifier,
        full_onnx_session,
        head_onnx_session,
    )


def main() -> None:
    """Run all inference benchmarks."""
    args = parse_args()

    dataframe = pd.read_csv(
        TEST_DATASET_PATH,
    )

    if TEXT_COLUMN not in dataframe.columns:
        msg = f"Dataset does not contain required column: {TEXT_COLUMN}"
        raise ValueError(msg)

    if dataframe[TEXT_COLUMN].isna().any():
        msg = f"Dataset column {TEXT_COLUMN} contains null values."
        raise ValueError(msg)

    texts = dataframe[TEXT_COLUMN].astype(str).tolist()

    (
        pipeline,
        tfidf,
        classifier,
        full_onnx_session,
        head_onnx_session,
    ) = load_resources()

    full_input_name = full_onnx_session.get_inputs()[0].name

    head_input_name = head_onnx_session.get_inputs()[0].name

    def sklearn_inference(
        text: str,
    ) -> int:
        """Run complete sklearn pipeline inference."""
        prediction = pipeline.predict(
            [text],
        )

        return int(
            prediction[0],
        )

    def hybrid_inference(
        text: str,
    ) -> int:
        """Run sklearn TF-IDF followed by ONNX classifier."""
        features = tfidf.transform([text]).astype(np.float32).toarray()

        predictions, _ = head_onnx_session.run(
            None,
            {
                head_input_name: features,
            },
        )

        return int(
            predictions[0],
        )

    def full_onnx_inference(
        text: str,
    ) -> int:
        """Run complete ONNX inference pipeline."""
        onnx_input = np.asarray(
            [[text]],
            dtype=object,
        )

        predictions, _ = full_onnx_session.run(
            None,
            {
                full_input_name: (onnx_input),
            },
        )

        return int(
            predictions[0],
        )

    # Ensure baseline classifier is initialized.
    _ = classifier.classes_

    print(f"Benchmark configuration: runs={args.runs}, warmup={args.warmup}")

    print(
        "ONNX Runtime configuration: "
        f"intra_op_threads="
        f"{ONNX_INTRA_OP_THREADS}, "
        f"inter_op_threads="
        f"{ONNX_INTER_OP_THREADS}, "
        "execution_mode=ORT_SEQUENTIAL"
    )

    print()

    print("Benchmarking sklearn...")
    sklearn_results = benchmark_backend(
        sklearn_inference,
        texts,
        args.runs,
        args.warmup,
    )

    print("Benchmarking hybrid...")
    hybrid_results = benchmark_backend(
        hybrid_inference,
        texts,
        args.runs,
        args.warmup,
    )

    print("Benchmarking full ONNX...")
    full_onnx_results = benchmark_backend(
        full_onnx_inference,
        texts,
        args.runs,
        args.warmup,
    )

    sklearn_mean = sklearn_results["mean_ms"]

    report = {
        "configuration": {
            "runs": args.runs,
            "warmup": args.warmup,
            "dataset": str(TEST_DATASET_PATH),
            "request_mode": ("single-text"),
            "onnx_provider": ("CPUExecutionProvider"),
            "onnx_runtime": {
                "intra_op_num_threads": (ONNX_INTRA_OP_THREADS),
                "inter_op_num_threads": (ONNX_INTER_OP_THREADS),
                "execution_mode": ("ORT_SEQUENTIAL"),
            },
        },
        "backends": {
            "sklearn": (sklearn_results),
            "hybrid": (hybrid_results),
            "full_onnx": (full_onnx_results),
        },
        "speedup_vs_sklearn": {
            "hybrid_mean": (sklearn_mean / hybrid_results["mean_ms"]),
            "full_onnx_mean": (sklearn_mean / full_onnx_results["mean_ms"]),
        },
        "artifact_sizes_bytes": {
            "joblib": (JOBLIB_PATH.stat().st_size),
            "full_onnx": (FULL_ONNX_PATH.stat().st_size),
            "onnx_classifier_head": (ONNX_HEAD_PATH.stat().st_size),
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== INFERENCE BENCHMARK ===")

    for (
        backend_name,
        result,
    ) in report["backends"].items():
        print()
        print(backend_name.upper())

        print(f"  mean: {result['mean_ms']:.4f} ms")
        print(f"  p50:  {result['p50_ms']:.4f} ms")
        print(f"  p95:  {result['p95_ms']:.4f} ms")
        print(f"  p99:  {result['p99_ms']:.4f} ms")
        print(f"  min:  {result['min_ms']:.4f} ms")
        print(f"  max:  {result['max_ms']:.4f} ms")
        print(f"  stdev: {result['stdev_ms']:.4f} ms")
        print(f"  req/s estimate: {result['estimated_requests_per_second']:.2f}")

    print()
    print("=== SPEEDUP VS SKLEARN ===")

    print(f"Hybrid:    {report['speedup_vs_sklearn']['hybrid_mean']:.3f}x")

    print(f"Full ONNX: {report['speedup_vs_sklearn']['full_onnx_mean']:.3f}x")

    print()
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
