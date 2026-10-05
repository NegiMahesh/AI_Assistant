from __future__ import annotations

import inspect
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
#
# Keep a global override for easy tuning, but use smaller model-specific
# contexts by default so the fast CPU models do not reserve unnecessary
# context memory.
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))


def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value:
            try:
                return max(512, int(value))
            except ValueError:
                pass
    return default


MODEL_CONTEXT = {
    FAST_MODEL: _env_int("FAST_NUM_CTX", default=2048),
    GENERAL_MODEL: _env_int("GENERAL_NUM_CTX", default=3072),
    COMPLEX_MODEL: _env_int("COMPLEX_NUM_CTX", default=4096),
    CODING_MODEL: _env_int("CODING_NUM_CTX", default=4096),
    VISION_MODEL: _env_int("VISION_NUM_CTX", default=4096),
}


MODEL_MAX_TOKENS = {
    FAST_MODEL: _env_int("FAST_NUM_PREDICT", default=192),
    GENERAL_MODEL: _env_int("GENERAL_NUM_PREDICT", default=384),
    COMPLEX_MODEL: _env_int("COMPLEX_NUM_PREDICT", default=768),
    CODING_MODEL: _env_int("CODING_NUM_PREDICT", default=512),
    VISION_MODEL: _env_int("VISION_NUM_PREDICT", default=512),
}


def get_model_options(
    model_name: str,
    overrides: dict | None = None,
) -> dict:
    options = {
        "num_ctx": MODEL_CONTEXT.get(model_name, DEFAULT_NUM_CTX),
        "num_predict": MODEL_MAX_TOKENS.get(model_name),
    }

    if overrides:
        options.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )

    return options


THINK_MODE = os.getenv("OLLAMA_THINK_MODE", "auto").lower().strip()
MODEL_THINK = {
    FAST_MODEL: False,
    GENERAL_MODEL: False,
    COMPLEX_MODEL: True,
}
CHAT_SUPPORTS_THINK = "think" in inspect.signature(ollama.chat).parameters


def get_think(model_name: str):
    if THINK_MODE in {"true", "on", "1"}:
        return True
    if THINK_MODE in {"false", "off", "0"}:
        return False
    return MODEL_THINK.get(model_name)

# Keep models warm briefly. Ollama can evict models when memory is needed,
# while avoiding an expensive reload on every model switch.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60s")

# Keep the small/medium models warm for fast switching, but never keep more
# than one large model resident at the same time on a CPU-only machine.
LARGE_MODELS = {
    COMPLEX_MODEL,
    CODING_MODEL,
    VISION_MODEL,
}
MAX_LARGE_RESIDENT = max(
    1,
    int(os.getenv("MAX_LARGE_RESIDENT", "1")),
)

ACTIVE_MODEL: str | None = None
MODEL_SWITCH_COUNT = 0
LARGE_MODEL_EVICTION_COUNT = 0

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
    """Unload one model without deleting it from the Ollama model store."""
    try:
        ollama.chat(
            model=model_name,
            messages=[],
            keep_alive=0,
            options=get_model_options(model_name),
        )
        print(f"Ollama model unloaded: {model_name}")
    except Exception as error:
        print(f"Could not unload {model_name}: {error}")


def get_loaded_models() -> list[dict]:
    """Return currently resident Ollama models for performance monitoring."""
    try:
        response = ollama.ps()
    except Exception as error:
        return [{"error": str(error)}]

    loaded = []

    for model in getattr(response, "models", []) or []:
        loaded.append(
            {
                "model": getattr(model, "model", None),
                "size": getattr(model, "size", None),
                "size_vram": getattr(model, "size_vram", None),
                "expires_at": getattr(model, "expires_at", None),
                "context_length": getattr(model, "context_length", None),
            }
        )

    return loaded


def enforce_large_model_limit(target_model: str) -> None:
    """Keep at most MAX_LARGE_RESIDENT large models resident."""
    global LARGE_MODEL_EVICTION_COUNT

    if target_model not in LARGE_MODELS:
        return

    loaded = get_loaded_models()
    if any(item.get("error") for item in loaded):
        return

    resident_large = [
        item.get("model")
        for item in loaded
        if item.get("model") in LARGE_MODELS
    ]

    # The target itself may already be resident, so no eviction is needed.
    other_large = [
        model
        for model in resident_large
        if model and model != target_model
    ]

    allowed_others = max(0, MAX_LARGE_RESIDENT - 1)

    for model in other_large[: max(0, len(other_large) - allowed_others)]:
        unload_model(model)
        LARGE_MODEL_EVICTION_COUNT += 1


def prepare_model(model_name: str) -> None:
    """Track the active model and enforce large-model residency limits."""
    global ACTIVE_MODEL, MODEL_SWITCH_COUNT

    if model_name in LARGE_MODELS:
        enforce_large_model_limit(model_name)

    if ACTIVE_MODEL != model_name:
        print(f"Ollama active model: {model_name}")
        ACTIVE_MODEL = model_name
        MODEL_SWITCH_COUNT += 1


def get_keep_alive(model_name: str) -> str:
    return KEEP_ALIVE


def chat(
    model_name: str,
    messages: Iterable[dict],
    options_override: dict | None = None,
):
    prepare_model(model_name)

    kwargs = {
        "model": model_name,
        "messages": list(messages),
        "keep_alive": get_keep_alive(model_name),
        "options": get_model_options(model_name, options_override),
    }

    think = get_think(model_name)
    if CHAT_SUPPORTS_THINK and think is not None:
        kwargs["think"] = think

    return ollama.chat(**kwargs)


def stream_chat(
    model_name: str,
    messages: Iterable[dict],
    options_override: dict | None = None,
):
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
        kwargs = {
            "model": model_name,
            "messages": list(messages),
            "stream": True,
            "keep_alive": get_keep_alive(model_name),
            "options": get_model_options(model_name, options_override),
        }

        think = get_think(model_name)
        if CHAT_SUPPORTS_THINK and think is not None:
            kwargs["think"] = think

        response = ollama.chat(**kwargs)

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
        "model_options": get_model_options(LAST_PERFORMANCE["model"])
        if LAST_PERFORMANCE["model"] else get_model_options(GENERAL_MODEL),
        "model_context": MODEL_CONTEXT,
        "default_num_ctx": DEFAULT_NUM_CTX,
        "model_max_tokens": MODEL_MAX_TOKENS,
        "think_mode": THINK_MODE,
        "chat_supports_think": CHAT_SUPPORTS_THINK,
        "model_think": get_think(LAST_PERFORMANCE["model"])
        if LAST_PERFORMANCE["model"] else None,
        "large_models": sorted(LARGE_MODELS),
        "max_large_resident": MAX_LARGE_RESIDENT,
        "large_model_eviction_count": LARGE_MODEL_EVICTION_COUNT,
        "loaded_models": get_loaded_models(),
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
        "model_options": get_model_options(ACTIVE_MODEL)
        if ACTIVE_MODEL else get_model_options(GENERAL_MODEL),
        "model_context": MODEL_CONTEXT,
        "default_num_ctx": DEFAULT_NUM_CTX,
        "model_max_tokens": MODEL_MAX_TOKENS,
        "keep_alive": KEEP_ALIVE,
        "large_models": sorted(LARGE_MODELS),
        "max_large_resident": MAX_LARGE_RESIDENT,
        "large_model_eviction_count": LARGE_MODEL_EVICTION_COUNT,
        "loaded_models": get_loaded_models(),
    }
