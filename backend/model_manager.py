from __future__ import annotations

import os
import time
from typing import Iterable

import ollama

FAST_MODEL = os.getenv("FAST_MODEL", "qwen3:0.6b")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "qwen3:1.7b")
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "qwen3:4b")
CODING_MODEL = os.getenv("CODING_MODEL", "qwen2.5-coder:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:4b")

ALL_MODELS = [
    FAST_MODEL,
    GENERAL_MODEL,
    COMPLEX_MODEL,
    CODING_MODEL,
    VISION_MODEL,
]

# Ollama already decides how much work to place on CPU/GPU. The old
# num_gpu=0 option explicitly disabled GPU acceleration.
MODEL_OPTIONS = {
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
}

# Keep models warm briefly. Ollama can evict models when memory is needed,
# while avoiding an expensive reload on every model switch.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60s")
ACTIVE_MODEL: str | None = None
MODEL_SWITCH_COUNT = 0

LAST_PERFORMANCE = {
    "model": None,
    "first_token_ms": None,
    "total_ms": None,
    "load_duration_ms": None,
    "prompt_eval_duration_ms": None,
    "eval_duration_ms": None,
    "prompt_tokens": None,
    "generated_tokens": None,
    "tokens_per_second": None,
    "error": None,
}


def unload_model(model_name: str) -> None:
    """Explicitly unload a model when the caller really needs to free memory."""
    try:
        ollama.chat(
            model=model_name,
            messages=[],
            keep_alive=0,
            options=MODEL_OPTIONS,
        )
        print(f"Ollama model unloaded: {model_name}")
    except Exception as error:
        print(f"Could not unload {model_name}: {error}")


def prepare_model(model_name: str) -> None:
    """Mark a model active without forcing the previous model out of memory."""
    global ACTIVE_MODEL, MODEL_SWITCH_COUNT

    if ACTIVE_MODEL != model_name:
        print(f"Ollama active model: {model_name}")
        ACTIVE_MODEL = model_name
        MODEL_SWITCH_COUNT += 1


def get_keep_alive(model_name: str) -> str:
    return KEEP_ALIVE


def chat(model_name: str, messages: Iterable[dict]):
    prepare_model(model_name)

    return ollama.chat(
        model=model_name,
        messages=list(messages),
        keep_alive=get_keep_alive(model_name),
        options=MODEL_OPTIONS,
    )


def stream_chat(model_name: str, messages: Iterable[dict]):
    prepare_model(model_name)

    started = time.perf_counter()
    first_token_at = None
    final_chunk = {}

    LAST_PERFORMANCE.update(
        {
            "model": model_name,
            "first_token_ms": None,
            "total_ms": None,
            "load_duration_ms": None,
            "prompt_eval_duration_ms": None,
            "eval_duration_ms": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "tokens_per_second": None,
            "error": None,
        }
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=list(messages),
            stream=True,
            keep_alive=get_keep_alive(model_name),
            options=MODEL_OPTIONS,
        )

        for chunk in response:
            final_chunk = chunk
            try:
                content = chunk.get("message", {}).get("content", "")
            except AttributeError:
                content = ""

            # Record first response chunk as soon as Ollama starts
            # producing output, even if its text content is empty.
            if first_token_at is None and (
                content or chunk.get("done") is True
            ):
                first_token_at = time.perf_counter()
                LAST_PERFORMANCE["first_token_ms"] = round(
                    (first_token_at - started) * 1000, 2
                )

            yield chunk

        _update_stream_metrics(final_chunk)

    except Exception as error:
        LAST_PERFORMANCE["error"] = str(error)
        raise

    finally:
        # A disconnected/aborted client may close the generator before the
        # final Ollama chunk reaches us. Always preserve elapsed time.
        LAST_PERFORMANCE["total_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )


def _update_stream_metrics(final_chunk: dict) -> None:
    load_duration = final_chunk.get("load_duration")
    prompt_eval_duration = final_chunk.get("prompt_eval_duration")
    eval_duration = final_chunk.get("eval_duration")
    prompt_tokens = final_chunk.get("prompt_eval_count")
    generated_tokens = final_chunk.get("eval_count")

    if load_duration is not None:
        LAST_PERFORMANCE["load_duration_ms"] = round(
            load_duration / 1_000_000, 2
        )
    if prompt_eval_duration is not None:
        LAST_PERFORMANCE["prompt_eval_duration_ms"] = round(
            prompt_eval_duration / 1_000_000, 2
        )
    if eval_duration is not None:
        LAST_PERFORMANCE["eval_duration_ms"] = round(
            eval_duration / 1_000_000, 2
        )
    if prompt_tokens is not None:
        LAST_PERFORMANCE["prompt_tokens"] = prompt_tokens
    if generated_tokens is not None:
        LAST_PERFORMANCE["generated_tokens"] = generated_tokens

    if eval_duration and generated_tokens:
        LAST_PERFORMANCE["tokens_per_second"] = round(
            generated_tokens / (eval_duration / 1_000_000_000), 2
        )


def performance_status():
    return {
        **LAST_PERFORMANCE,
        "active_model": ACTIVE_MODEL,
        "model_switch_count": MODEL_SWITCH_COUNT,
        "keep_alive": KEEP_ALIVE,
        "model_options": MODEL_OPTIONS,
    }


def model_status():
    return {
        "fast": FAST_MODEL,
        "general": GENERAL_MODEL,
        "complex": COMPLEX_MODEL,
        "coding": CODING_MODEL,
        "vision": VISION_MODEL,
        "active_model": ACTIVE_MODEL,
        "gpu_auto": True,
        "model_options": MODEL_OPTIONS,
        "keep_alive": KEEP_ALIVE,
    }
