"""
Phase 19.8 — Router performance benchmark.

This benchmark measures only deterministic routing/model-selection overhead.
It does NOT call Ollama, so it is safe to run repeatedly on a CPU-only laptop.

Run from the project root:
    python backend/router/benchmark_router.py

Optional:
    $env:ROUTER_BENCHMARK_ITERATIONS="5000"
    python backend/router/benchmark_router.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from router.ai_router import route_input, select_model  # noqa: E402


CASES = (
    "Hello",
    "What is RAM?",
    "How does virtual memory work?",
    "Explain RISC vs CISC in detail and compare their advantages",
    "Can you explain this Python error and tell me how to fix it?",
    "Hello, can you explain what a pointer is?",
    "calculate 25 + 15",
    "search for the latest Python news",
    "open chrome",
    "who am i",
    "what objects are there",
)


def benchmark(iterations: int = 1000) -> tuple[float, float, float]:
    timings = []

    for _ in range(iterations):
        start = time.perf_counter()

        for text in CASES:
            routing = route_input(text)
            if routing.get("route") == "ai":
                select_model(
                    text,
                    route="ai",
                    intent=routing.get("intent", "chat"),
                )

        timings.append((time.perf_counter() - start) * 1000)

    return (
        statistics.mean(timings),
        statistics.median(timings),
        max(timings),
    )


def main() -> None:
    iterations = int(os.getenv("ROUTER_BENCHMARK_ITERATIONS", "1000"))

    if iterations <= 0:
        raise ValueError("ROUTER_BENCHMARK_ITERATIONS must be greater than 0")

    mean_ms, median_ms, max_ms = benchmark(iterations)

    print("Phase 19.8 Router Benchmark")
    print("===========================")
    print(f"Cases per iteration : {len(CASES)}")
    print(f"Iterations          : {iterations}")
    print(f"Average batch time  : {mean_ms:.3f} ms")
    print(f"Median batch time   : {median_ms:.3f} ms")
    print(f"Maximum batch time  : {max_ms:.3f} ms")
    print()
    print("This measures routing/model-selection logic only.")
    print("Ollama model loading and token generation are not included.")


if __name__ == "__main__":
    main()
