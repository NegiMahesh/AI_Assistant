from faster_whisper import WhisperModel

print("Loading speech recognition model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Speech model loaded!")


def transcribe_audio(audio_file):
    segments, info = model.transcribe(
        audio_file,
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()