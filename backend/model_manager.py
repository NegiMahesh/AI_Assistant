from __future__ import annotations

import os
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
    global ACTIVE_MODEL

    if ACTIVE_MODEL != model_name:
        print(f"Ollama active model: {model_name}")
        ACTIVE_MODEL = model_name


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

    response = ollama.chat(
        model=model_name,
        messages=list(messages),
        stream=True,
        keep_alive=get_keep_alive(model_name),
        options=MODEL_OPTIONS,
    )

    for chunk in response:
        yield chunk


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
