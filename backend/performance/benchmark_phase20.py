"""
Phase 20.8 — Final CPU/RAM/latency benchmark.

Run from the project root on Windows:
    python backend/performance/benchmark_phase20.py

Optional:
    python backend/performance/benchmark_phase20.py --iterations 2

This benchmark uses the real router/model manager and samples Ollama's
Windows process CPU and RAM while each response is generated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_manager import (  # noqa: E402
    performance_status,
    stream_chat,
)
from router.ai_router import (  # noqa: E402
    get_response_token_budget,
    route_input,
    select_model,
)


CASES = (
    "Hello",
    "What is RAM?",
    "How does virtual memory work?",
)


def ollama_process_stats() -> dict:
    """Return summed Ollama CPU time and working-set memory on Windows."""
    script = (
        "Get-Process | "
        "Where-Object { $_.ProcessName -like 'ollama*' } | "
        "Select-Object ProcessName,CPU,WorkingSet64 | "
        "ConvertTo-Json -Compress"
    )

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return {"cpu_seconds": 0.0, "ram_bytes": 0, "processes": 0}

        parsed = json.loads(result.stdout)

        if isinstance(parsed, dict):
            parsed = [parsed]

        cpu_seconds = sum(float(item.get("CPU") or 0.0) for item in parsed)
        ram_bytes = sum(int(item.get("WorkingSet64") or 0) for item in parsed)

        return {
            "cpu_seconds": cpu_seconds,
            "ram_bytes": ram_bytes,
            "processes": len(parsed),
        }

    except Exception:
        return {"cpu_seconds": 0.0, "ram_bytes": 0, "processes": 0}


class ResourceSampler:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def _run(self) -> None:
        previous = ollama_process_stats()
        previous_time = time.perf_counter()

        while not self.stop_event.wait(self.interval):
            current = ollama_process_stats()
            now = time.perf_counter()

            elapsed = max(now - previous_time, 1e-6)
            logical_cpus = max(os.cpu_count() or 1, 1)
            cpu_percent = (
                max(0.0, current["cpu_seconds"] - previous["cpu_seconds"])
                / elapsed
                / logical_cpus
                * 100.0
            )

            self.samples.append(
                {
                    "cpu_percent": cpu_percent,
                    "ram_mb": current["ram_bytes"] / (1024 * 1024),
                    "processes": current["processes"],
                }
            )

            previous = current
            previous_time = now

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "avg_cpu_percent": None,
                "peak_cpu_percent": None,
                "peak_ram_mb": None,
                "avg_ram_mb": None,
                "ollama_processes": None,
            }

        cpus = [sample["cpu_percent"] for sample in self.samples]
        rams = [sample["ram_mb"] for sample in self.samples]

        return {
            "avg_cpu_percent": round(sum(cpus) / len(cpus), 2),
            "peak_cpu_percent": round(max(cpus), 2),
            "peak_ram_mb": round(max(rams), 2),
            "avg_ram_mb": round(sum(rams) / len(rams), 2),
            "ollama_processes": max(
                sample["processes"] for sample in self.samples
            ),
        }


def run_case(prompt: str) -> dict:
    routing = route_input(prompt)

    if routing.get("route") != "ai":
        return {
            "prompt": prompt,
            "route": routing.get("route"),
            "intent": routing.get("intent"),
            "model": None,
            "error": "Benchmark case did not route to AI.",
        }

    intent = routing.get("intent", "chat")
    model = select_model(
        prompt,
        route="ai",
        intent=intent,
    )
    token_budget = get_response_token_budget(
        prompt,
        intent=intent,
    )

    sampler = ResourceSampler()
    chunks: list[str] = []

    started = time.perf_counter()
    sampler.start()

    try:
        for chunk in stream_chat(
            model,
            [{"role": "user", "content": prompt}],
            options_override={"num_predict": token_budget},
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                chunks.append(content)
    finally:
        sampler.stop()

    wall_clock_ms = (time.perf_counter() - started) * 1000
    metrics = dict(performance_status())
    resources = sampler.summary()

    return {
        "prompt": prompt,
        "intent": intent,
        "model": model,
        "token_budget": token_budget,
        "wall_clock_ms": round(wall_clock_ms, 2),
        "first_token_ms": metrics.get("first_token_ms"),
        "model_load_ms": metrics.get("load_duration_ms"),
        "prompt_eval_ms": metrics.get("prompt_eval_duration_ms"),
        "generation_ms": metrics.get("eval_duration_ms"),
        "generated_tokens": metrics.get("generated_tokens"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "response_chars": len("".join(chunks)),
        **resources,
        "loaded_models": [
            item.get("model")
            for item in metrics.get("loaded_models", [])
            if item.get("model")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Repeat the complete case set this many times.",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    print("Phase 20.8 Final Performance Benchmark")
    print("======================================")
    print(f"Cases      : {len(CASES)}")
    print(f"Iterations : {args.iterations}")
    print()

    all_results = []

    for iteration in range(1, args.iterations + 1):
        print(f"Iteration {iteration}")

        for prompt in CASES:
            result = run_case(prompt)
            all_results.append(result)

            print(f"  {result['intent']} -> {result['model']}")
            print(f"    First token : {result.get('first_token_ms')} ms")
            print(f"    Total       : {result.get('wall_clock_ms')} ms")
            print(f"    Generated   : {result.get('generated_tokens')} tokens")
            print(f"    Tokens/sec  : {result.get('tokens_per_second')}")
            print(f"    Avg CPU     : {result.get('avg_cpu_percent')}%")
            print(f"    Peak CPU    : {result.get('peak_cpu_percent')}%")
            print(f"    Peak RAM    : {result.get('peak_ram_mb')} MB")
            print(f"    Loaded      : {result.get('loaded_models')}")
            print()

    successful = [item for item in all_results if "error" not in item]

    if successful:
        total_times = [
            item["wall_clock_ms"]
            for item in successful
            if item.get("wall_clock_ms") is not None
        ]
        first_tokens = [
            item["first_token_ms"]
            for item in successful
            if item.get("first_token_ms") is not None
        ]

        print("Overall")
        print("-------")
        if total_times:
            print(f"Average total latency : {sum(total_times) / len(total_times):.2f} ms")
        if first_tokens:
            print(f"Average first token  : {sum(first_tokens) / len(first_tokens):.2f} ms")

    print()
    print("Run this benchmark after the optimized version is loaded.")
    print("It measures Ollama CPU/RAM, latency, token speed, routing, and residency.")
    print("Do not compare cold and warm runs as if they were identical workloads.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
