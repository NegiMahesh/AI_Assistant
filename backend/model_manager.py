from __future__ import annotations
import ollama

FAST_MODEL = "qwen3:0.6b"
GENERAL_MODEL = "qwen3:1.7b"
COMPLEX_MODEL = "qwen3:4b"
CODING_MODEL = "qwen2.5-coder:3b"
VISION_MODEL = "gemma3:4b"
ALL_MODELS = [FAST_MODEL, GENERAL_MODEL, COMPLEX_MODEL, CODING_MODEL, VISION_MODEL]
CPU_OPTIONS = {"num_gpu": 0, "num_ctx": 4096}
GENERAL_KEEP_ALIVE = "45s"
SPECIAL_KEEP_ALIVE = 0
ACTIVE_MODEL: str | None = None

def unload_model(model_name: str):
    try:
        ollama.chat(model=model_name, messages=[], keep_alive=0, options=CPU_OPTIONS)
        print(f"Ollama model unloaded: {model_name}")
    except Exception as error:
        print(f"Could not unload {model_name}: {error}")

def prepare_model(model_name: str):
    global ACTIVE_MODEL
    if ACTIVE_MODEL == model_name:
        return
    if ACTIVE_MODEL:
        unload_model(ACTIVE_MODEL)
    ACTIVE_MODEL = model_name
    print(f"Ollama active model: {model_name}")

def get_keep_alive(model_name: str):
    return GENERAL_KEEP_ALIVE if model_name == GENERAL_MODEL else SPECIAL_KEEP_ALIVE

def chat(model_name: str, messages):
    global ACTIVE_MODEL
    prepare_model(model_name)
    keep_alive = get_keep_alive(model_name)
    response = ollama.chat(model=model_name, messages=messages, keep_alive=keep_alive, options=CPU_OPTIONS)
    if keep_alive == 0:
        ACTIVE_MODEL = None
    return response

def stream_chat(model_name: str, messages):
    global ACTIVE_MODEL
    prepare_model(model_name)
    keep_alive = get_keep_alive(model_name)
    response = ollama.chat(model=model_name, messages=messages, stream=True, keep_alive=keep_alive, options=CPU_OPTIONS)
    try:
        for chunk in response:
            yield chunk
    finally:
        if keep_alive == 0:
            ACTIVE_MODEL = None

def model_status():
    return {
        "fast": FAST_MODEL, "general": GENERAL_MODEL, "complex": COMPLEX_MODEL,
        "coding": CODING_MODEL, "vision": VISION_MODEL, "active_model": ACTIVE_MODEL,
        "cpu_only": True, "general_keep_alive": GENERAL_KEEP_ALIVE,
        "special_keep_alive": SPECIAL_KEEP_ALIVE,
    }
