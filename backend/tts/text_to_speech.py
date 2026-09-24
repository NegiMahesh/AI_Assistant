from piper import PiperVoice
from pathlib import Path
import wave


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "en_US-lessac-medium.onnx"
OUTPUT_PATH = BASE_DIR / "output.wav"


# =========================================================
# LOAD TTS MODEL
# =========================================================

print("Loading TTS model...")

try:
    voice = PiperVoice.load(MODEL_PATH)
    print("TTS model loaded!")

except Exception as error:
    voice = None
    print(f"TTS model loading failed: {error}")


# =========================================================
# TEXT TO SPEECH
# =========================================================

def speak(text: str):

    if voice is None:
        raise RuntimeError("TTS model is not loaded.")

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    text = text.strip()

    try:

        with wave.open(str(OUTPUT_PATH), "wb") as wav_file:

            voice.synthesize_wav(
                text,
                wav_file
            )

        return OUTPUT_PATH

    except Exception as error:

        print(f"TTS synthesis error: {error}")
        raise