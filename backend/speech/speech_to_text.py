import numpy as np
import torch
import av
from qwen_asr import Qwen3ASRModel

# Qwen3-ASR-1.7B uses the Transformers backend on CPU. Float32 is selected
# for broad CPU compatibility; this is memory-intensive and slower than GPU use.
STT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

print(f"Loading speech recognition model ({STT_MODEL_ID}, CPU float32)...")

model = Qwen3ASRModel.from_pretrained(
    STT_MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    max_inference_batch_size=1,
    max_new_tokens=256,
)

print("Speech model loaded (Qwen3-ASR-1.7B, CPU)!")


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
    waveform = _load_audio_as_16khz_mono(audio_file)

    results = model.transcribe(
        audio=(waveform, 16000),
        language=None,
        return_time_stamps=False,
    )

    if not results:
        return ""

    result = results[0]
    detected_language = getattr(result, "language", "unknown")
    text = getattr(result, "text", "").strip()

    print(f"Speech language detected by Qwen3-ASR: {detected_language}")
    print(f"Qwen3-ASR transcription: {text}")

    return text
