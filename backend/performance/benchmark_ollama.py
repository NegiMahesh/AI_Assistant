"""
Phase 20.1 — Direct Ollama latency benchmark.

This bypasses FastAPI and the React UI and measures the selected Ollama
model directly. It runs two requests so cold-start and warm-request
latency can be compared.

Examples:
    python backend/performance/benchmark_ollama.py
    python backend/performance/benchmark_ollama.py --model qwen3:1.7b
    python backend/performance/benchmark_ollama.py --model qwen3:0.6b --iterations 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_manager import (  # noqa: E402
    FAST_MODEL,
    GENERAL_MODEL,
    COMPLEX_MODEL,
    CODING_MODEL,
    performance_status,
    stream_chat,
)


DEFAULT_PROMPT = "Explain what RAM is in simple terms."


def run_once(model: str, prompt: str) -> dict:
    started = time.perf_counter()
    chunks = []

    for chunk in stream_chat(
        model,
        [{"role": "user", "content": prompt}],
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:
            chunks.append(content)

    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics = dict(performance_status())
    metrics["wall_clock_ms"] = round(elapsed_ms, 2)
    metrics["response_chars"] = len("".join(chunks))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=GENERAL_MODEL,
        help="Single Ollama model to benchmark.",
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="Run a model sequence in the same Python process, e.g. "
        "--sequence qwen3:1.7b qwen3:0.6b qwen3:1.7b",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of requests. 2 is recommended for cold vs warm comparison.",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    known_models = {
        FAST_MODEL,
        GENERAL_MODEL,
        COMPLEX_MODEL,
        CODING_MODEL,
    }

    print("Phase 20.1 Ollama Benchmark")
    print("===========================")
    models = args.sequence or [args.model]

    print(f"Models     : {" -> ".join(models)}")
    print(f"Iterations : {args.iterations}")
    print(f"Prompt     : {args.prompt}")
    print()

    for model in models:
        if model not in known_models:
            print(f"Warning: {model} is not one of the configured AI models.")

        for index in range(1, args.iterations + 1):
            metrics = run_once(model, args.prompt)

            print(f"Request {index} [{model}]")
            print(f"  Wall clock       : {metrics['wall_clock_ms']} ms")
            print(f"  First token      : {metrics['first_token_ms']} ms")
            print(f"  Model load       : {metrics['load_duration_ms']} ms")
            print(f"  Prompt eval      : {metrics['prompt_eval_duration_ms']} ms")
            print(f"  Generation       : {metrics['eval_duration_ms']} ms")
            print(f"  Prompt tokens    : {metrics['prompt_tokens']}")
            print(f"  Generated tokens : {metrics['generated_tokens']}")
            print(f"  Tokens/sec       : {metrics['tokens_per_second']}")
            print(f"  Response chars   : {metrics['response_chars']}")
            print(f"  Model switches   : {metrics['model_switch_count']}")
            print(f"  Think mode       : {metrics.get('model_think')}")
            print()

    print("Keep-alive:", performance_status()["keep_alive"])
    print("Context:", performance_status()["model_options"]["num_ctx"])
    print("Use Request 1 vs Request 2 to identify cold-start/model-load cost.")


if __name__ == "__main__":
    main()
