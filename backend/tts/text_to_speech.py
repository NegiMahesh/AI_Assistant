from piper import PiperVoice
from pathlib import Path
import wave


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "en_US-lessac-medium.onnx"

ESPEAK_DATA_DIR = (
    BASE_DIR
    / "venv"
    / "Lib"
    / "site-packages"
    / "piper"
    / "espeak-ng-data"
)

OUTPUT_PATH = BASE_DIR / "output.wav"


print("Loading TTS model...")

voice = PiperVoice.load(
    MODEL_PATH,
    espeak_data_dir=ESPEAK_DATA_DIR
)

print("TTS model loaded!")


def speak(text: str):
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    return OUTPUT_PATH