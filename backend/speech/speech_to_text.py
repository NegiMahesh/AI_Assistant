import time

import numpy as np
import torch
import av
from qwen_asr import Qwen3ASRModel

# Qwen3-ASR-1.7B is memory-intensive on CPU. Keeping the generation budget
# bounded helps avoid very long inference for short voice commands.
STT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

print(f"Loading speech recognition model ({STT_MODEL_ID}, CPU float32)...", flush=True)

model = Qwen3ASRModel.from_pretrained(
    STT_MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    max_inference_batch_size=1,
    max_new_tokens=128,
)

print("Speech model loaded (Qwen3-ASR-1.7B, CPU)!", flush=True)


def _load_audio_as_16khz_mono(audio_file):
    """Decode browser audio (including WebM) to mono float32 PCM at 16 kHz."""
    chunks = []

    with av.open(audio_file) as container:
        if not container.streams.audio:
            raise ValueError("The uploaded file does not contain an audio stream.")

        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=16000,
        )

        for frame in container.decode(audio=0):
            for output_frame in resampler.resample(frame):
                chunks.append(
                    output_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                )

        for output_frame in (resampler.resample(None) or []):
            chunks.append(
                output_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            )

    if not chunks:
        raise ValueError("No decodable audio samples were found in the recording.")

    return np.concatenate(chunks).astype(np.float32, copy=False)


def transcribe_audio(audio_file):
    print("[STT] Decoding recorded audio...", flush=True)
    decode_started = time.perf_counter()
    waveform = _load_audio_as_16khz_mono(audio_file)
    audio_seconds = len(waveform) / 16000.0
    decode_elapsed = time.perf_counter() - decode_started

    print(
        f"[STT] Audio decoded: {audio_seconds:.1f}s, "
        f"{len(waveform)} samples (decode {decode_elapsed:.2f}s). "
        "Running Qwen inference on CPU...",
        flush=True,
    )

    inference_started = time.perf_counter()
    results = model.transcribe(
        audio=(waveform, 16000),
        language=None,
        return_time_stamps=False,
    )
    inference_elapsed = time.perf_counter() - inference_started
    print(f"[STT] Qwen inference returned in {inference_elapsed:.1f}s.", flush=True)

    if not results:
        print("[STT] Qwen returned no transcription results.", flush=True)
        return ""

    result = results[0]
    if isinstance(result, dict):
        detected_language = result.get("language", "unknown")
        text = str(result.get("text", "")).strip()
    else:
        detected_language = getattr(result, "language", "unknown")
        text = str(getattr(result, "text", "")).strip()

    print(f"Speech language detected by Qwen3-ASR: {detected_language}", flush=True)
    print(f"Qwen3-ASR transcription: {text}", flush=True)

    return text
