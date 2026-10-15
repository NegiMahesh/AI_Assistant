import os

from faster_whisper import WhisperModel

# "small" is more accurate than "base" for general speech, while int8 keeps
# CPU memory use lower. Override with AI_ASSISTANT_STT_MODEL=base if the
# larger model is too slow for this computer.
STT_MODEL_SIZE = os.getenv("AI_ASSISTANT_STT_MODEL", "small").strip().lower() or "small"

print(f"Loading speech recognition model ({STT_MODEL_SIZE}, CPU int8)...")

model = WhisperModel(
    STT_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

print(f"Speech model loaded ({STT_MODEL_SIZE})!")


def transcribe_audio(audio_file):
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    text = "".join(segment.text for segment in segments).strip()

    if info is not None:
        print(
            f"Speech language detected: {info.language} "
            f"(probability={info.language_probability:.2f})"
        )

    return text