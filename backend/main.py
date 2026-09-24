from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ollama
import os
import tempfile
import re
import ast
import operator
import subprocess
import platform
import json

from datetime import datetime, timedelta
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from speech.speech_to_text import transcribe_audio
from tts.text_to_speech import speak
from router.ai_router import route_input, select_model

from model_manager import (
    chat as model_chat,
    stream_chat as model_stream_chat,
    GENERAL_MODEL,
    model_status,
)
from file_reader.file_reader import read_file, get_file_info
from tools.web_search import web_search

from memory.memory import (
    save_memory,
    search_memories,
    get_recent_memories,
    get_all_memories,
    delete_memory,
    delete_all_memories,
    get_memory_count,
    process_memory,
    get_relevant_memories,
    build_memory_context,
)

# =========================================================
# PHASE 16.6 — COMMAND / ACTION SYSTEM
# =========================================================

from actions.action_engine import handle_action

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Fast AI Assistant",
    description=(
        "AI Assistant with LLM, Voice, YOLO, Face Recognition, "
        "Intelligent Memory, File Reader, Conversation History, "
        "Context-Aware Tools, Streaming and Command Actions"
    ),
    version="2.3",
)


# =========================================================
# CORS
# =========================================================

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# VISION MODELS — LAZY LOADING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "yolo11n.pt")
VISION_MODEL = None
FACE_APP = None


def get_vision_models():
    """Load heavy vision models only when vision is actually used."""
    global VISION_MODEL, FACE_APP

    if VISION_MODEL is None:
        print("Loading YOLO model...")
        VISION_MODEL = YOLO(MODEL_PATH)
        print("YOLO model loaded!")

    if FACE_APP is None:
        print("Loading InsightFace model...")
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                face_providers = ["CPUExecutionProvider"]
        except Exception:
            face_providers = ["CPUExecutionProvider"]

        FACE_APP = FaceAnalysis(
            name="buffalo_l",
            providers=face_providers,
        )
        FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
        print(f"InsightFace model loaded ({face_providers[0]}).")

    return VISION_MODEL, FACE_APP


# =========================================================
# FACE DATABASE
# =========================================================

FACE_DATABASE_FILE = os.path.join(BASE_DIR, "face_database.npz")

FACE_DATABASE = {}


# =========================================================
# SETTINGS
# =========================================================

DEFAULT_CONFIDENCE = 0.40

YOLO_IMAGE_SIZE = 640

MAX_DETECTIONS = 30

FACE_MATCH_THRESHOLD = 0.45

SUPPORTED_FILE_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}

MAX_FILE_CONTEXT = 12000

MAX_HISTORY_MESSAGES = 12


# =========================================================
# REQUEST MODELS
# =========================================================


class ChatMessage(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    message: str

    vision: dict | None = None

    file_context: str | None = None

    history: list[ChatMessage] = Field(default_factory=list)


class SpeakRequest(BaseModel):

    text: str


class MemoryRequest(BaseModel):

    memory: str

    category: str = "general"


class FileQuestionRequest(BaseModel):

    question: str


# =========================================================
# FACE EMBEDDING
# =========================================================


def normalize_embedding(embedding):

    if embedding is None:

        return None

    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)

    if norm < 1e-10:

        return None

    return embedding / norm


# =========================================================
# FACE DATABASE LOAD
# =========================================================


def load_face_database():

    global FACE_DATABASE

    FACE_DATABASE = {}

    if not os.path.exists(FACE_DATABASE_FILE):

        print("No saved face database found.")

        return

    try:

        data = np.load(FACE_DATABASE_FILE, allow_pickle=True)

        names = data["names"]

        embeddings = data["embeddings"]

        for name, embedding in zip(names, embeddings):

            normalized = normalize_embedding(embedding)

            if normalized is not None:

                FACE_DATABASE[str(name)] = normalized

        print(f"Loaded {len(FACE_DATABASE)} saved face(s).")

    except Exception as error:

        print("Could not load face database:", error)

        FACE_DATABASE = {}


# =========================================================
# FACE DATABASE SAVE
# =========================================================


def save_face_database():

    try:

        names = np.array(list(FACE_DATABASE.keys()), dtype=object)

        embeddings = np.array(list(FACE_DATABASE.values()), dtype=np.float32)

        np.savez(FACE_DATABASE_FILE, names=names, embeddings=embeddings)

        print(f"Face database saved: " f"{len(FACE_DATABASE)} face(s)")

    except Exception as error:

        print("Could not save face database:", error)


load_face_database()


# =========================================================
# HOME
# =========================================================


@app.get("/")
def home():

    return {"message": "AI Assistant is running!", "status": "online"}


# =========================================================
# STATUS
# =========================================================


@app.get("/status")
def status():

    return {
        "assistant": "online",
        "llm": model_status()["general"],
        "yolo": "yolo11n",
        "face_recognition": "InsightFace",
        "speech_recognition": "enabled",
        "tts": "enabled",
        "router": "enabled",
        "weather": "enabled (Open-Meteo + forecasts)",
        "calendar": "enabled (local events.json)",
        "memory": "Phase 13C intelligent memory",
        "memory_count": get_memory_count(),
        "saved_faces": len(FACE_DATABASE),
        "file_reader": "enabled",
        "supported_files": list(SUPPORTED_FILE_EXTENSIONS),
        "conversation_history": "enabled",
        "context_aware_tools": "enabled",
        "streaming": "enabled",
        "command_actions": "enabled",
        "max_history_messages": MAX_HISTORY_MESSAGES,
    }


# =========================================================
# CALCULATOR
# =========================================================


def calculate_expression(expression: str):

    expression = expression.lower().strip()

    prefixes = [
        r"^(please\s+)?calculate\s+",
        r"^what\s+is\s+",
        r"^what's\s+",
        r"^solve\s+",
        r"^compute\s+",
        r"^how\s+much\s+is\s+",
    ]

    for pattern in prefixes:

        expression = re.sub(pattern, "", expression)

    expression = expression.strip()

    replacements = [
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\btimes\b", "*"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bover\b", "/"),
        (r"\bmodulo\b", "%"),
        (r"\bmod\b", "%"),
    ]

    for pattern, value in replacements:

        expression = re.sub(pattern, value, expression)

    percentage_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", expression
    )

    if percentage_match:

        percentage = float(percentage_match.group(1))

        number = float(percentage_match.group(2))

        return (percentage / 100) * number

    percentage_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*percent\s+of\s+(\d+(?:\.\d+)?)", expression
    )

    if percentage_match:

        percentage = float(percentage_match.group(1))

        number = float(percentage_match.group(2))

        return (percentage / 100) * number

    expression = expression.rstrip("?").strip().replace("^", "**")

    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):

        raise ValueError("Invalid mathematical expression")

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):

        if isinstance(node, ast.Expression):

            return evaluate(node.body)

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):

                return node.value

            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):

            left = evaluate(node.left)

            right = evaluate(node.right)

            operation = allowed_operators.get(type(node.op))

            if operation is None:

                raise ValueError("Operator not allowed")

            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:

                raise ValueError("Division by zero")

            if isinstance(node.op, ast.Pow):

                if abs(right) > 100:

                    raise ValueError("Power value too large")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):

            value = evaluate(node.operand)

            operation = allowed_operators.get(type(node.op))

            if operation is None:

                raise ValueError("Operator not allowed")

            return operation(value)

        raise ValueError("Invalid expression")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree)


def looks_like_calculation(text: str):

    text = text.lower().strip()

    if re.search(r"\b(calculate|calculation|compute|solve)\b", text):

        return True

    if re.search(r"\d+(?:\.\d+)?\s*%\s*of\s*\d+(?:\.\d+)?", text):

        return True

    if re.search(r"\d+(?:\.\d+)?\s*percent\s+of\s+\d+(?:\.\d+)?", text):

        return True

    cleaned = re.sub(r"^(what\s+is|what's|how\s+much\s+is|please)\s+", "", text)

    cleaned = cleaned.rstrip("?").strip()

    if re.fullmatch(r"\d+(?:\.\d+)?\s*[+\-*/%^]\s*\d+(?:\.\d+)?", cleaned):

        return True

    if re.fullmatch(r"[\d\s+\-*/%^().]+", cleaned):

        if any(
            symbol in cleaned
            for symbol in (
                "+",
                "-",
                "*",
                "/",
                "%",
                "^",
            )
        ):

            return True

    if re.search(
        r"\d+(?:\.\d+)?\s+"
        r"(plus|minus|times|multiplied by|divided by|over)"
        r"\s+\d+(?:\.\d+)?",
        text,
    ):

        return True

    return False


def format_calculation_result(result):

    if isinstance(result, float):

        if result.is_integer():

            return str(int(result))

        return str(round(result, 10))

    return str(result)


# =========================================================
# LEGACY APP CONTROL
# =========================================================
# Kept so your existing router remains compatible.
# Phase 16.6 uses the dedicated action engine first.
# =========================================================


def open_application(app_name: str):

    app_name = app_name.lower().strip()

    if "chrome" in app_name:

        if platform.system() == "Windows":

            subprocess.Popen("start chrome", shell=True)

            return "Opening Chrome."

    if "calculator" in app_name or "calc" in app_name:

        if platform.system() == "Windows":

            subprocess.Popen("calc.exe")

            return "Opening Calculator."

    if "notepad" in app_name:

        if platform.system() == "Windows":

            subprocess.Popen("notepad.exe")

            return "Opening Notepad."

    return "I don't know how to open " + app_name + " yet."


def close_application(app_name: str):

    app_name = app_name.lower().strip()

    if "chrome" in app_name:

        if platform.system() == "Windows":

            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/IM",
                    "chrome.exe",
                ],
                capture_output=True,
            )

            return "Chrome closed."

    if "calculator" in app_name or "calc" in app_name:

        if platform.system() == "Windows":

            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/IM",
                    "CalculatorApp.exe",
                ],
                capture_output=True,
            )

            return "Calculator closed."

    if "notepad" in app_name:

        if platform.system() == "Windows":

            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/IM",
                    "notepad.exe",
                ],
                capture_output=True,
            )

            return "Notepad closed."

    return "I don't know how to close " + app_name + " yet."


# =========================================================
# MEMORY
# =========================================================


def format_relevant_memories(user_input: str):

    try:

        memories = get_relevant_memories(user_input, 8)

    except Exception as error:

        print("Relevant memory search error:", error)

        return ""

    if not memories:

        return ""

    formatted = []

    for memory in memories:

        if isinstance(memory, dict):

            memory_text = (
                memory.get("memory")
                or memory.get("text")
                or memory.get("content")
                or memory.get("value")
            )

            category = memory.get("category") or "general"

            if memory_text:

                formatted.append(f"- {memory_text} " f"(category: {category})")

        elif isinstance(memory, str):

            formatted.append(f"- {memory}")

    if not formatted:

        return ""

    return "\n".join(formatted)


def is_personal_memory_question(text: str):

    text = text.lower().strip()

    patterns = [
        r"\bwhat('?s| is) my name\b",
        r"\bwho am i\b",
        r"\bdo you know my name\b",
        r"\bwhat do you call me\b",
        r"\bwhat is my favourite\b",
        r"\bwhat is my favorite\b",
        r"\bwhat('?s| is) my preferred\b",
        r"\bwhat programming language do i like\b",
        r"\bwhat programming language is my favorite\b",
        r"\bwhat is my favorite programming language\b",
        r"\bwhat is my favourite programming language\b",
        r"\bwhat do i like\b",
        r"\bwhat are my preferences\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


# =========================================================
# WEATHER
# =========================================================


def get_weather_code_description(code):

    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }

    return descriptions.get(code, "unknown conditions")


def geocode_location(location):

    location = (location or "").strip()

    if not location:

        location = "Dehradun"

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={quote(location)}"
        "&count=1"
        "&language=en"
        "&format=json"
    )

    request = Request(url, headers={"User-Agent": "Fast-AI-Assistant/2.3"})

    with urlopen(request, timeout=8) as response:

        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results") or []

    if not results:

        return None

    result = results[0]

    return {
        "name": result.get("name") or location,
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "country": result.get("country") or "",
        "admin1": result.get("admin1") or "",
    }


def format_place_name(place):

    place_name = place.get("name") or "Unknown location"

    if place.get("admin1") and place["admin1"] != place_name:

        place_name += f", {place['admin1']}"

    if place.get("country"):

        place_name += f", {place['country']}"

    return place_name


def get_weather(location="Dehradun"):

    place = geocode_location(location)

    if not place:

        return f"I couldn't find the location " f"'{location}'."

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={place['latitude']}"
        f"&longitude={place['longitude']}"
        "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "apparent_temperature,"
        "weather_code,"
        "wind_speed_10m"
        "&timezone=auto"
    )

    request = Request(url, headers={"User-Agent": "Fast-AI-Assistant/2.3"})

    with urlopen(request, timeout=8) as response:

        data = json.loads(response.read().decode("utf-8"))

    current = data.get("current", {})

    temperature = current.get("temperature_2m")

    feels_like = current.get("apparent_temperature")

    humidity = current.get("relative_humidity_2m")

    wind = current.get("wind_speed_10m")

    code = current.get("weather_code")

    parts = [
        f"Weather in "
        f"{format_place_name(place)}: "
        f"{get_weather_code_description(code)}"
    ]

    if temperature is not None:

        parts.append(f"temperature {temperature}°C")

    if feels_like is not None:

        parts.append(f"feels like {feels_like}°C")

    if humidity is not None:

        parts.append(f"humidity {humidity}%")

    if wind is not None:

        parts.append(f"wind {wind} km/h")

    return ", ".join(parts) + "."


def get_weather_forecast(location="Dehradun", days=2):

    place = geocode_location(location)

    if not place:

        return f"I couldn't find the location " f"'{location}'."

    forecast_days = max(2, min(days, 16))

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={place['latitude']}"
        f"&longitude={place['longitude']}"
        "&daily="
        "weather_code,"
        "temperature_2m_max,"
        "temperature_2m_min,"
        "precipitation_probability_max"
        f"&forecast_days={forecast_days}"
        "&timezone=auto"
    )

    request = Request(url, headers={"User-Agent": "Fast-AI-Assistant/2.3"})

    with urlopen(request, timeout=8) as response:

        data = json.loads(response.read().decode("utf-8"))

    daily = data.get("daily", {})

    dates = daily.get("time") or []

    codes = daily.get("weather_code") or []

    highs = daily.get("temperature_2m_max") or []

    lows = daily.get("temperature_2m_min") or []

    rain_probability = daily.get("precipitation_probability_max") or []

    if not dates:

        return f"I couldn't get a forecast for " f"{format_place_name(place)}."

    target_index = 1 if len(dates) > 1 else 0

    day_label = "Tomorrow" if target_index == 1 else "Today"

    code = codes[target_index] if target_index < len(codes) else None

    result = (
        f"{day_label} in "
        f"{format_place_name(place)}: "
        f"{get_weather_code_description(code)}"
    )

    if target_index < len(highs):

        result += f", high {highs[target_index]}°C"

    if target_index < len(lows):

        result += f", low {lows[target_index]}°C"

    if (
        target_index < len(rain_probability)
        and rain_probability[target_index] is not None
    ):

        result += f", precipitation probability " f"{rain_probability[target_index]}%"

    return result + "."


def is_weather_question(text):

    text = text.lower().strip()

    patterns = [
        r"\bweather\b",
        r"\btemperature\b",
        r"\bhow hot is it\b",
        r"\bhow cold is it\b",
        r"\bis it raining\b",
        r"\bwill it rain\b",
        r"\bforecast\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def extract_weather_location(text):

    text = text.strip()

    patterns = [
        r"\bweather\s+(?:in|at|for)\s+(.+?)(?:\?|$)",
        r"\btemperature\s+(?:in|at|for)\s+(.+?)(?:\?|$)",
        r"\bforecast\s+(?:in|at|for)\s+(.+?)(?:\?|$)",
        r"\bin\s+([A-Za-z][A-Za-z .'-]{1,60})\s*$",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:

            location = match.group(1).strip(" .?!,")

            if location:

                return location

    return "Dehradun"


def history_contains_weather(history):

    if not history:

        return False

    for item in reversed(history[-8:]):

        if isinstance(item, ChatMessage):

            content = item.content

        elif isinstance(item, dict):

            content = str(item.get("content", ""))

        else:

            continue

        content = content.lower().strip()

        if not content:

            continue

        if is_weather_question(content):

            return True

        if re.search(r"\b(weather|forecast|temperature|" r"raining|rain)\b", content):

            return True

    return False


def extract_weather_location_from_history(history):

    if not history:

        return None

    for item in reversed(history[-8:]):

        if isinstance(item, ChatMessage):

            content = item.content

        elif isinstance(item, dict):

            content = str(item.get("content", ""))

        else:

            continue

        content = content.strip()

        if not content:

            continue

        explicit_match = re.search(
            r"\b(?:weather|temperature|forecast)"
            r"\s+(?:in|at|for)\s+"
            r"(.+?)(?:\?|$)",
            content,
            flags=re.IGNORECASE,
        )

        if explicit_match:

            candidate = explicit_match.group(1).strip(" .?!,")

            if candidate:

                return candidate

        explicit_match = re.search(
            r"\b(?:in|at|for)\s+" r"([A-Za-z][A-Za-z .'-]{1,60})" r"(?:\?|$)",
            content,
            flags=re.IGNORECASE,
        )

        if explicit_match:

            candidate = explicit_match.group(1).strip(" .?!,")

            if candidate:

                return candidate

    return None


def is_weather_followup(text, history):

    if not history_contains_weather(history):

        return False

    text = text.lower().strip()

    patterns = [
        r"^what\s+about\s+tomorrow\??$",
        r"^what\s+about\s+today\??$",
        r"^and\s+tomorrow\??$",
        r"^and\s+today\??$",
        r"^tomorrow\??$",
        r"^today\??$",
        r"^what\s+about\s+the\s+next\s+day\??$",
        r"^will\s+it\s+rain\??$",
        r"^is\s+it\s+going\s+to\s+rain\??$",
        r"^how\s+about\s+tomorrow\??$",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def weather_response(text, history=None):

    history = history or []

    if is_weather_followup(text, history):

        location = extract_weather_location_from_history(history) or "Dehradun"

        if "tomorrow" in (text.lower()):

            try:

                return get_weather_forecast(location, days=2)

            except Exception as error:

                print("Weather forecast error:", error)

                return "I couldn't fetch " "tomorrow's forecast right now."

        try:

            return get_weather(location)

        except Exception as error:

            print("Weather tool error:", error)

            return "I couldn't fetch the weather right now."

    location = extract_weather_location(text)

    text_lower = text.lower()

    try:

        if "tomorrow" in text_lower:

            return get_weather_forecast(location, days=2)

        return get_weather(location)

    except Exception as error:

        print("Weather tool error:", error)

        return (
            "I couldn't fetch the weather right now. "
            "Please check your internet connection."
        )


# =========================================================
# CALENDAR
# =========================================================

CALENDAR_FILE = "events.json"


def _load_calendar_events():

    if not os.path.exists(CALENDAR_FILE):

        return []

    try:

        with open(CALENDAR_FILE, "r", encoding="utf-8") as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        return []

    except Exception as error:

        print("Calendar load error:", error)

        return []


def _save_calendar_events(events):

    with open(CALENDAR_FILE, "w", encoding="utf-8") as file:

        json.dump(events, file, indent=4, ensure_ascii=False)


def _calendar_datetime_text(value):

    if not value:

        return ""

    try:

        dt = datetime.fromisoformat(value)

        return dt.strftime("%A, %d %B %Y at %I:%M %p")

    except Exception:

        return str(value)


def _parse_calendar_datetime(text):

    raw = text.lower().strip()

    now = datetime.now()

    target_date = None

    if re.search(r"\btomorrow\b", raw):

        target_date = (now + timedelta(days=1)).date()

    elif re.search(r"\btoday\b", raw):

        target_date = now.date()

    else:

        iso_match = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", raw)

        if iso_match:

            try:

                target_date = datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()

            except ValueError:

                pass

        if target_date is None:

            date_match = re.search(
                r"\b(\d{1,2})\s+"
                r"(january|february|march|april|may|june|july|"
                r"august|september|october|november|december)"
                r"(?:\s+(20\d{2}))?\b",
                raw,
            )

            if date_match:

                day = int(date_match.group(1))

                month_name = date_match.group(2)

                year = int(date_match.group(3) or now.year)

                try:

                    target_date = datetime.strptime(
                        f"{day} {month_name} {year}", "%d %B %Y"
                    ).date()

                except ValueError:

                    pass

    if target_date is None:

        target_date = now.date()

    target_time = None

    time_match = re.search(
        r"\b(?:at|@)\s*" r"(\d{1,2})(?::(\d{2}))?\s*" r"(a\.?m\.?|p\.?m\.?)\b",
        raw,
        flags=re.IGNORECASE,
    )

    if time_match:

        hour = int(time_match.group(1))

        minute = int(time_match.group(2) or 0)

        meridiem = time_match.group(3).lower().replace(".", "")

        if meridiem == "pm" and hour != 12:

            hour += 12

        elif meridiem == "am" and hour == 12:

            hour = 0

        if 0 <= hour <= 23 and 0 <= minute <= 59:

            target_time = (hour, minute)

    else:

        twenty_four_hour_match = re.search(
            r"\b(?:at|@)\s*" r"([01]?\d|2[0-3]):([0-5]\d)\b", raw
        )

        if twenty_four_hour_match:

            target_time = (
                int(twenty_four_hour_match.group(1)),
                int(twenty_four_hour_match.group(2)),
            )

    if target_time is None:

        target_time = (now.hour, now.minute)

    return datetime.combine(target_date, datetime.min.time()).replace(
        hour=target_time[0], minute=target_time[1], second=0, microsecond=0
    )


def _extract_event_title(text):

    title = text.strip()

    # -----------------------------------------------------
    # REMOVE COMMAND PREFIX
    # -----------------------------------------------------

    title = re.sub(
        r"^\s*(please\s+)?" r"(add|create|schedule|set|put|save|remember|book|plan)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # REMOVE OPTIONAL EVENT WORD
    # -----------------------------------------------------

    title = re.sub(
        r"^\s*(an?\s+)?" r"(event|appointment|reminder|task)" r"(?:\s+called)?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # REMOVE DATE
    # -----------------------------------------------------

    title = re.sub(
        r"\s+(?:on\s+)?"
        r"(?:today|tomorrow|tonight|"
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}\s+"
        r"(?:january|february|march|april|may|june|july|"
        r"august|september|october|november|december)"
        r"(?:\s+20\d{2})?)"
        r".*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # REMOVE 12-HOUR TIME
    # -----------------------------------------------------

    title = re.sub(
        r"\s+(?:at|@)\s*\d{1,2}" r"(?::\d{2})?\s*" r"(?:a\.?m\.?|p\.?m\.?)" r".*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # -----------------------------------------------------
    # REMOVE 24-HOUR TIME
    # -----------------------------------------------------

    title = re.sub(r"\s+(?:at|@)\s*[012]?\d:[0-5]\d.*$", "", title, flags=re.IGNORECASE)

    # -----------------------------------------------------
    # CLEAN EXTRA WORDS
    # -----------------------------------------------------

    title = re.sub(r"\s+", " ", title).strip(" .,!?:;-")

    return title or "Untitled event"


def add_calendar_event(text):

    events = _load_calendar_events()

    event_datetime = _parse_calendar_datetime(text)

    title = _extract_event_title(text)

    event_id = (
        max(
            [
                int(event.get("id", 0))
                for event in events
                if str(event.get("id", "")).isdigit()
            ],
            default=0,
        )
        + 1
    )

    event = {
        "id": event_id,
        "title": title,
        "datetime": event_datetime.isoformat(),
        "created_at": datetime.now().isoformat(),
    }

    events.append(event)

    events.sort(key=lambda item: item.get("datetime", ""))

    _save_calendar_events(events)

    return event


def get_calendar_events_for_date(target_date):

    events = _load_calendar_events()

    result = []

    for event in events:

        try:

            event_dt = datetime.fromisoformat(event.get("datetime", ""))

            if event_dt.date() == target_date:

                result.append(event)

        except Exception:

            continue

    result.sort(key=lambda item: item.get("datetime", ""))

    return result


def get_today_calendar_events():

    return get_calendar_events_for_date(datetime.now().date())


def get_tomorrow_calendar_events():

    return get_calendar_events_for_date((datetime.now() + timedelta(days=1)).date())


def get_upcoming_calendar_events(limit=10):

    now = datetime.now()

    events = _load_calendar_events()

    upcoming = []

    for event in events:

        try:

            event_dt = datetime.fromisoformat(event.get("datetime", ""))

            if event_dt >= now:

                upcoming.append(event)

        except Exception:

            continue

    upcoming.sort(key=lambda item: item.get("datetime", ""))

    return upcoming[:limit]


def delete_calendar_event(event_id):

    events = _load_calendar_events()

    original_count = len(events)

    events = [event for event in events if str(event.get("id")) != str(event_id)]

    if len(events) == original_count:

        return False

    _save_calendar_events(events)

    return True


def is_calendar_question(text):

    text = text.lower().strip()

    patterns = [
        # -------------------------------------------------
        # DATE / DAY
        # -------------------------------------------------
        r"\bwhat date is today\b",
        r"\bwhat is today's date\b",
        r"\bwhat's today's date\b",
        r"\bwhat day is today\b",
        r"\bwhat date are we\b",
        r"\bwhat day are we\b",
        r"\bwhat is the date\b",
        r"\btoday's date\b",
        # -------------------------------------------------
        # ADD / CREATE / SCHEDULE EVENTS
        # -------------------------------------------------
        r"\b(add|create|schedule|set|put|save|book|plan)\b"
        r".*\b"
        r"(meeting|event|appointment|class|reminder|call|task|"
        r"gym|gym time|workout|workout time|study|study time|"
        r"exam|interview|doctor|doctor appointment|"
        r"lunch|dinner|breakfast|practice|session)\b",
        r"\b(add|create|schedule|set|put|save|book|plan)\b"
        r".*\b"
        r"(today|tomorrow|tonight|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"\b",
        # -------------------------------------------------
        # DATE/TIME + EVENT STYLE REQUESTS
        # -------------------------------------------------
        r"\b(today|tomorrow|tonight)\b"
        r".*\b(at|@)\s*\d{1,2}"
        r"(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b",
        r"\b(at|@)\s*\d{1,2}"
        r"(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b"
        r".*\b(today|tomorrow|tonight)\b",
        # -------------------------------------------------
        # CALENDAR LOOKUPS
        # -------------------------------------------------
        r"\bwhat do i have\b",
        r"\bwhat have i got\b",
        r"\bwhat's on my calendar\b",
        r"\bwhat is on my calendar\b",
        r"\bshow my calendar\b",
        r"\bshow my events\b",
        r"\bshow upcoming events\b",
        r"\bshow my upcoming events\b",
        r"\bupcoming events\b",
        r"\bmy upcoming events\b",
        r"\bwhat are my events\b",
        r"\bmy schedule\b",
        r"\bwhat is my schedule\b",
        r"\bwhat's my schedule\b",
        r"\bwhat do i have tomorrow\b",
        r"\bwhat do i have today\b",
        r"\bwhat's happening tomorrow\b",
        r"\bwhat is happening tomorrow\b",
        r"\bwhat's happening today\b",
        r"\bwhat is happening today\b",
        r"\bwhat events do i have\b",
        # -------------------------------------------------
        # GENERAL CALENDAR TERMS
        # -------------------------------------------------
        r"\bcalendar\b",
        r"\bschedule\b",
        r"\bappointment\b",
        r"\breminder\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def history_contains_calendar(history):

    if not history:

        return False

    for item in reversed(history[-8:]):

        if isinstance(item, ChatMessage):

            content = item.content

        elif isinstance(item, dict):

            content = str(item.get("content", ""))

        else:

            continue

        content = content.lower().strip()

        if not content:

            continue

        if is_calendar_question(content):

            return True

        if re.search(
            r"\b(calendar|schedule|meeting|" r"appointment|event|reminder)\b", content
        ):

            return True

    return False


def is_calendar_followup(text, history):

    if not history_contains_calendar(history):

        return False

    text = text.lower().strip()

    patterns = [
        r"^what\s+about\s+tomorrow\??$",
        r"^what\s+about\s+today\??$",
        r"^and\s+tomorrow\??$",
        r"^and\s+today\??$",
        r"^tomorrow\??$",
        r"^today\??$",
        r"^what\s+about\s+the\s+next\s+day\??$",
        r"^what\s+do\s+i\s+have\s+tomorrow\??$",
        r"^what\s+do\s+i\s+have\s+today\??$",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def calendar_response(text, history=None):

    history = history or []

    text_lower = text.lower().strip()

    if is_calendar_followup(text, history):

        if "tomorrow" in text_lower or "next day" in text_lower:

            events = get_tomorrow_calendar_events()

            if not events:

                return "You have no events " "scheduled for tomorrow."

            lines = []

            for index, event in enumerate(events, 1):

                lines.append(
                    f"{index}. "
                    f"{event.get('title', 'Untitled event')} "
                    f"— "
                    f"{_calendar_datetime_text(event.get('datetime'))}"
                )

            return "Your events for tomorrow:\n" + "\n".join(lines)

        events = get_today_calendar_events()

        if not events:

            return "You have no events " "scheduled for today."

        lines = []

        for index, event in enumerate(events, 1):

            lines.append(
                f"{index}. "
                f"{event.get('title', 'Untitled event')} "
                f"— "
                f"{_calendar_datetime_text(event.get('datetime'))}"
            )

        return "Your events for today:\n" + "\n".join(lines)

    if (
        re.search(r"\bwhat date is today\b", text_lower)
        or re.search(r"\bwhat is today's date\b", text_lower)
        or re.search(r"\bwhat's today's date\b", text_lower)
        or re.search(r"\bwhat day is today\b", text_lower)
        or re.search(r"\bwhat date are we\b", text_lower)
        or re.search(r"\bwhat day are we\b", text_lower)
        or re.search(r"\bwhat is the date\b", text_lower)
        or re.search(r"\btoday's date\b", text_lower)
    ):

        return f"Today is " f"{datetime.now().strftime('%A, %d %B %Y')}."

    if re.search(r"\b(add|create|schedule|set|put|save|book|plan)\b", text_lower):

        try:

            event = add_calendar_event(text)

            return (
                f"Added '{event['title']}' for "
                f"{_calendar_datetime_text(event['datetime'])}."
            )

        except Exception as error:

            print("Calendar add error:", error)

            return (
                "I couldn't add that event. " "Please include the event name and time."
            )

    if "tomorrow" in text_lower or "next day" in text_lower:

        events = get_tomorrow_calendar_events()

        if not events:

            return "You have no events " "scheduled for tomorrow."

        lines = []

        for index, event in enumerate(events, 1):

            lines.append(
                f"{index}. "
                f"{event.get('title', 'Untitled event')} "
                f"— "
                f"{_calendar_datetime_text(event.get('datetime'))}"
            )

        return "Your events for tomorrow:\n" + "\n".join(lines)

    if "today" in text_lower or "today's" in text_lower:

        events = get_today_calendar_events()

        if not events:

            return "You have no events " "scheduled for today."

        lines = []

        for index, event in enumerate(events, 1):

            lines.append(
                f"{index}. "
                f"{event.get('title', 'Untitled event')} "
                f"— "
                f"{_calendar_datetime_text(event.get('datetime'))}"
            )

        return "Your events for today:\n" + "\n".join(lines)

    events = get_upcoming_calendar_events()

    if not events:

        return "You have no upcoming events."

    lines = []

    for index, event in enumerate(events, 1):

        lines.append(
            f"{index}. "
            f"{event.get('title', 'Untitled event')} "
            f"— "
            f"{_calendar_datetime_text(event.get('datetime'))}"
        )

    return "Your upcoming events:\n" + "\n".join(lines)


# =========================================================
# VISION
# =========================================================


def is_face_question(text: str):

    text = text.lower().strip()

    patterns = [
        r"\bwho am i\b",
        r"\bwho is in front of me\b",
        r"\bwho is that\b",
        r"\bdo you recognize me\b",
        r"\bdo you know me\b",
        r"\brecognize me\b",
        r"\bwhose face is this\b",
        r"\bwhat is my name\b",
        r"\bwho do you see\b",
        r"\bwho are you looking at\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_object_question(text: str):

    text = text.lower().strip()

    patterns = [
        r"\bwhat is in front of me\b",
        r"\bwhat's in front of me\b",
        r"\bwhat do you see\b",
        r"\bwhat can you see\b",
        r"\bwhat is around me\b",
        r"\bwhat's around me\b",
        r"\bwhat objects are there\b",
        r"\bwhat objects do you see\b",
        r"\bdescribe what you see\b",
        r"\bwhat is in front\b",
        r"\bwhat's in front\b",
        r"\bwhat am i looking at\b",
        r"\bwhat is around\b",
        r"\bwhat's around\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def build_vision_description(vision):

    if not vision:

        return "I do not currently have " "camera information."

    detections = vision.get("detections", [])

    faces = vision.get("faces", [])

    parts = []

    if detections:

        object_counts = {}

        for detection in detections:

            name = detection.get("class", "unknown")

            object_counts[name] = object_counts.get(name, 0) + 1

        object_parts = []

        for name, count in object_counts.items():

            object_parts.append(f"{count} {name}")

        parts.append("I can see " + ", ".join(object_parts) + ".")

    else:

        parts.append("I do not currently see " "any detected objects.")

    if faces:

        known_people = []

        for face in faces:

            name = face.get("name", "Unknown")

            if name != "Unknown":

                known_people.append(name)

        if known_people:

            known_people = list(dict.fromkeys(known_people))

            parts.append("I recognize " + ", ".join(known_people) + ".")

        else:

            parts.append("I can see a face, " "but I do not recognize " "the person.")

    return " ".join(parts)


# =========================================================
# BUILD AI CONTEXT
# =========================================================


def build_ai_context(user_input: str, vision=None, file_context=None, history=None):

    context_parts = []

    if history:

        conversation_lines = []

        for item in history:

            if isinstance(item, ChatMessage):

                role = item.role.strip().lower()

                content = item.content.strip()

            elif isinstance(item, dict):

                role = str(item.get("role", "")).strip().lower()

                content = str(item.get("content", "")).strip()

            else:

                continue

            if role not in ("user", "assistant"):

                continue

            if not content:

                continue

            role_name = "USER" if role == "user" else "ASSISTANT"

            conversation_lines.append(f"{role_name}: {content}")

        conversation_lines = conversation_lines[-MAX_HISTORY_MESSAGES:]

        if conversation_lines:

            context_parts.append(
                "CONVERSATION HISTORY:\n" + "\n".join(conversation_lines)
            )

    memory_context = build_memory_context(user_input)

    if memory_context:

        context_parts.append("STORED USER MEMORY:\n" + memory_context)

    relevant_memory_text = format_relevant_memories(user_input)

    if relevant_memory_text:

        context_parts.append("RELEVANT SAVED USER FACTS:\n" + relevant_memory_text)

    if vision:

        context_parts.append(
            "CURRENT CAMERA INFORMATION:\n" + build_vision_description(vision)
        )

    if file_context:

        cleaned_file_context = file_context.strip()

        if cleaned_file_context:

            if len(cleaned_file_context) > MAX_FILE_CONTEXT:

                cleaned_file_context = (
                    cleaned_file_context[:MAX_FILE_CONTEXT] + "\n\n"
                    "[File content truncated because it is too large.]"
                )

            context_parts.append("UPLOADED FILE CONTENT:\n" + cleaned_file_context)

    if not context_parts:

        return user_input

    return (
        "You are a helpful personal AI assistant.\n\n"
        "CONVERSATION RULES:\n"
        "1. Use the recent conversation history to "
        "understand follow-up questions.\n"
        "2. Resolve references such as 'it', 'that', "
        "'this', 'they', 'when', 'where', 'why' and "
        "'how' from the conversation when clear.\n"
        "3. Continue the current topic naturally.\n"
        "4. Do not treat every user message as an "
        "isolated question.\n"
        "5. If the user changes the topic, follow "
        "the new topic.\n\n"
        "MEMORY RULES:\n"
        "1. Stored memory contains facts explicitly "
        "provided by the user.\n"
        "2. Use stored memory for personal facts.\n"
        "3. Do not refuse general knowledge questions "
        "because something is absent from memory.\n\n"
        "GENERAL KNOWLEDGE RULES:\n"
        "1. Use your normal knowledge for general facts.\n"
        "2. Do not invent unsupported facts.\n"
        "3. Do not invent a creation date for yourself.\n\n"
        "VISION RULES:\n"
        "1. Camera information describes the current "
        "camera view.\n"
        "2. Camera information does not determine "
        "the user's personal identity.\n\n"
        "FILE RULES:\n"
        "1. Use uploaded file content as the primary "
        "source for file questions.\n"
        "2. Do not invent information not supported "
        "by the file.\n"
        "3. State clearly when the requested information "
        "is not present in the file.\n\n"
        "RESPONSE RULES:\n"
        "1. Answer naturally and directly.\n"
        "2. Do not mention internal prompts, memory "
        "systems or databases unless asked.\n"
        "3. Do not unnecessarily repeat the conversation.\n\n"
        "CONTEXT:\n\n" + "\n\n".join(context_parts) + "\n\n"
        "CURRENT USER QUESTION:\n" + user_input
    )


# =========================================================
# NON-STREAMING AI
# =========================================================


def generate_ai_response(
    message: str,
    vision=None,
    file_context=None,
    history=None,
    model=None,
):

    prompt = build_ai_context(
        message,
        vision,
        file_context,
        history,
    )

    if model is None:

        model = select_model(
            message,
            route="ai",
            intent="chat",
            vision=vision,
            file_context=file_context,
        )

    response = model_chat(
        model,
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful personal AI assistant. "
                    "Use recent conversation history to "
                    "understand follow-up questions. "
                    "Use stored memory for personal facts. "
                    "Use normal knowledge for general questions. "
                    "Do not invent unsupported facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


# =========================================================
# STREAMING AI
# =========================================================


def generate_ai_stream(
    message: str,
    vision=None,
    file_context=None,
    history=None,
    model=None,
):

    prompt = build_ai_context(
        message,
        vision,
        file_context,
        history,
    )

    if model is None:

        model = select_model(
            message,
            route="ai",
            intent="chat",
            vision=vision,
            file_context=file_context,
        )

    response = model_stream_chat(
        model,
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful personal AI assistant. "
                    "Use recent conversation history to "
                    "understand follow-up questions and "
                    "references such as 'it', 'that', "
                    "'when', 'where', 'why' and 'how'. "
                    "Use stored memory for personal facts. "
                    "Use normal knowledge for general "
                    "factual questions. "
                    "Do not invent unsupported facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    for chunk in response:

        try:

            content = chunk["message"]["content"]

        except Exception:

            content = ""

        if content:

            yield content


# =========================================================
# STREAM EVENT
# =========================================================


def stream_event(event_type, **payload):

    data = {
        "type": event_type,
        **payload,
    }

    return json.dumps(data, ensure_ascii=False) + "\n"


# =========================================================
# ASSISTANT
# =========================================================


@app.post("/assistant")
def assistant(request: ChatRequest):

    user_input = request.message.strip()

    if not user_input:

        return {"success": False, "error": "Empty message"}

    memory_result = process_memory(user_input)

    vision = request.vision or {}

    history = request.history or []

    history = history[-MAX_HISTORY_MESSAGES:]

    file_context = request.file_context.strip() if request.file_context else ""

    if len(file_context) > MAX_FILE_CONTEXT:

        file_context = (
            file_context[:MAX_FILE_CONTEXT] + "\n\n"
            "[File content truncated because it is too large.]"
        )

    # =====================================================
    # WEATHER
    # =====================================================

    if is_weather_question(user_input) or is_weather_followup(user_input, history):

        return {
            "success": True,
            "route": "tool",
            "intent": "weather",
            "response": weather_response(user_input, history),
            "memory": memory_result,
        }

    # =====================================================
    # CALENDAR
    # =====================================================

    if is_calendar_question(user_input) or is_calendar_followup(user_input, history):

        try:

            return {
                "success": True,
                "route": "tool",
                "intent": "calendar",
                "response": calendar_response(user_input, history),
                "memory": memory_result,
            }

        except Exception as error:

            print("Calendar tool error:", error)

            return {
                "success": False,
                "route": "tool",
                "intent": "calendar",
                "response": "I couldn't access the local calendar.",
                "error": str(error),
                "memory": memory_result,
            }

    # =====================================================
    # WEB SEARCH
    # =====================================================

    routing = route_input(user_input)

    if routing.get("route") == "tool" and routing.get("intent") == "web_search":

        try:

            search_result = web_search(user_input, max_results=5)

            if not search_result.get("success"):
                return {
                    "success": False,
                    "route": "tool",
                    "intent": "web_search",
                    "response": "Web search failed.",
                    "error": search_result.get("error"),
                    "memory": memory_result,
                }

            results = search_result.get("results", [])

            if not results:
                return {
                    "success": True,
                    "route": "tool",
                    "intent": "web_search",
                    "response": "I couldn't find any useful search results.",
                    "results": [],
                    "memory": memory_result,
                }

            search_context = "\n\n".join(
                f"TITLE: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"SUMMARY: {item.get('snippet', '')}"
                for item in results
            )

            prompt = (
                "Use the following live web search results to answer the user's question.\n\n"
                "SEARCH RESULTS:\n" + search_context + "\n\n"
                "USER QUESTION:\n" + user_input + "\n\n"
                "Rules: Answer using the search results. Do not invent facts. "
                "Keep the answer clear and useful. Mention important sources when appropriate."
            )

            response = model_chat(
                GENERAL_MODEL,
                [
                    {
                        "role": "system",
                        "content": "You are a helpful local AI assistant. Answer using the supplied web search results. Do not invent information.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return {
                "success": True,
                "route": "tool",
                "intent": "web_search",
                "response": response["message"]["content"],
                "results": results,
                "memory": memory_result,
            }

        except Exception as error:

            print("Web search error:", error)

            return {
                "success": False,
                "route": "tool",
                "intent": "web_search",
                "response": "I couldn't complete the web search.",
                "error": str(error),
                "memory": memory_result,
            }

    # =====================================================
    # FACE
    # =====================================================

    if is_face_question(user_input):

        faces = vision.get("faces", [])

        if not faces:

            relevant_memory_text = format_relevant_memories(user_input)

            if relevant_memory_text:

                return {
                    "success": True,
                    "route": "memory",
                    "intent": "personal_memory",
                    "response": generate_ai_response(
                        user_input, {}, file_context, history
                    ),
                    "memory": memory_result,
                }

            return {
                "success": True,
                "route": "vision",
                "intent": "face_recognition",
                "response": "I don't currently see a face. "
                "Please make sure the camera is on "
                "and your face is visible.",
                "memory": memory_result,
            }

        known_faces = [
            face for face in faces if face.get("name", "Unknown") != "Unknown"
        ]

        if known_faces:

            names = list(dict.fromkeys([face.get("name") for face in known_faces]))

            if len(names) == 1:

                return {
                    "success": True,
                    "route": "vision",
                    "intent": "face_recognition",
                    "response": f"You are {names[0]}.",
                    "memory": memory_result,
                }

            return {
                "success": True,
                "route": "vision",
                "intent": "face_recognition",
                "response": "I recognize: " + ", ".join(names) + ".",
                "memory": memory_result,
            }

        return {
            "success": True,
            "route": "vision",
            "intent": "face_recognition",
            "response": "I can see your face, " "but I don't recognize you yet.",
            "memory": memory_result,
        }

    # =====================================================
    # OBJECT DETECTION
    # =====================================================

    if is_object_question(user_input):

        detections = vision.get("detections", [])

        if not detections:

            return {
                "success": True,
                "route": "vision",
                "intent": "object_detection",
                "response": "I don't currently detect " "any objects in front of me.",
                "memory": memory_result,
            }

        return {
            "success": True,
            "route": "vision",
            "intent": "object_detection",
            "response": build_vision_description(vision),
            "memory": memory_result,
        }

    # =====================================================
    # PERSONAL MEMORY
    # =====================================================

    if is_personal_memory_question(user_input):

        relevant_memory_text = format_relevant_memories(user_input)

        if relevant_memory_text:

            return {
                "success": True,
                "route": "memory",
                "intent": "personal_memory",
                "response": generate_ai_response(user_input, {}, file_context, history),
                "memory": memory_result,
            }

    # =====================================================
    # CALCULATOR
    # =====================================================

    if looks_like_calculation(user_input):

        try:

            result = calculate_expression(user_input)

            return {
                "success": True,
                "route": "command",
                "intent": "calculator",
                "response": format_calculation_result(result),
                "memory": memory_result,
            }

        except Exception as error:

            return {
                "success": False,
                "route": "command",
                "intent": "calculator",
                "response": "I could not calculate that.",
                "error": str(error),
                "memory": memory_result,
            }

    # =====================================================
    # PHASE 16.6 — COMMAND / ACTION SYSTEM
    # =====================================================

    try:

        action_response = handle_action(user_input)

    except Exception as error:

        print("Action engine error:", error)

        action_response = {
            "success": False,
            "action": "action_engine",
            "message": "I couldn't process that action.",
            "error": str(error),
        }

    if action_response is not None:

        return {
            "success": action_response.get("success", False),
            "route": "action",
            "intent": action_response.get("action", "command"),
            "response": action_response.get("message", "Action completed."),
            "action": action_response.get("action"),
            "memory": memory_result,
        }

    # =====================================================
    # ROUTER
    # =====================================================

    routing = route_input(user_input)

    route = routing.get("route", "ai")

    intent = routing.get("intent", "chat")

    # =====================================================
    # COMMAND ROUTE
    # =====================================================

    if route == "command":

        if intent == "calculator":

            try:

                result = calculate_expression(user_input)

                return {
                    "success": True,
                    "route": "command",
                    "intent": "calculator",
                    "response": format_calculation_result(result),
                    "memory": memory_result,
                }

            except Exception as error:

                return {
                    "success": False,
                    "route": "command",
                    "intent": "calculator",
                    "response": "I could not calculate that.",
                    "error": str(error),
                    "memory": memory_result,
                }

        if intent == "open_app":

            app_name = re.sub(
                r"^(open|launch|start|run)\s+", "", user_input, flags=re.IGNORECASE
            )

            return {
                "success": True,
                "route": "command",
                "intent": "open_app",
                "response": open_application(app_name),
                "memory": memory_result,
            }

        if intent == "close_app":

            app_name = re.sub(
                r"^(close|stop|exit|quit)\s+", "", user_input, flags=re.IGNORECASE
            )

            return {
                "success": True,
                "route": "command",
                "intent": "close_app",
                "response": close_application(app_name),
                "memory": memory_result,
            }

        return {
            "success": True,
            "route": "command",
            "intent": intent,
            "response": "This command is not implemented yet.",
            "memory": memory_result,
        }

    # =====================================================
    # AI ROUTE
    # =====================================================

    if route == "ai":

        selected_model = select_model(
            user_input,
            route="ai",
            intent=intent,
            vision=vision,
            file_context=file_context,
        )

        def event_generator():

            try:

                yield stream_event(
                    "start",
                    route="ai",
                    intent=intent,
                    model=selected_model,
                )

                for chunk in generate_ai_stream(
                    user_input,
                    vision,
                    file_context,
                    history,
                    model=selected_model,
                ):

                    yield stream_event(
                        "chunk",
                        content=chunk,
                    )

                yield stream_event(
                    "done",
                    model=selected_model,
                )

            except Exception as error:

                print("Streaming AI error:", error)

                yield stream_event(
                    "error",
                    message=str(error),
                )

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
        )

    # =====================================================
    # VISION ROUTE
    # =====================================================

    if route == "vision":

        if intent == "object_detection":

            detections = vision.get("detections", [])

            response = (
                build_vision_description(vision)
                if detections
                else "Object detection is active, "
                "but I don't currently see "
                "any detected objects."
            )

            return {
                "success": True,
                "route": "vision",
                "intent": "object_detection",
                "response": response,
                "memory": memory_result,
            }

        if intent == "face_recognition":

            faces = vision.get("faces", [])

            if not faces:

                response = (
                    "Face recognition is active, " "but I don't currently see a face."
                )

            else:

                known_names = [
                    face.get("name", "Unknown")
                    for face in faces
                    if face.get("name", "Unknown") != "Unknown"
                ]

                if known_names:

                    response = (
                        "I recognize " + ", ".join(dict.fromkeys(known_names)) + "."
                    )

                else:

                    response = (
                        "I can see a face, " "but I don't recognize " "the person."
                    )

            return {
                "success": True,
                "route": "vision",
                "intent": "face_recognition",
                "response": response,
                "memory": memory_result,
            }

        return {
            "success": True,
            "route": "vision",
            "intent": intent,
            "response": build_vision_description(vision),
            "memory": memory_result,
        }

    # =====================================================
    # FALLBACK STREAMING AI
    # =====================================================

    def fallback_generator():

        try:

            yield stream_event("start", route="ai", intent="chat")

            for chunk in generate_ai_stream(user_input, vision, file_context, history):

                yield stream_event("chunk", content=chunk)

            yield stream_event("done")

        except Exception as error:

            print("Fallback streaming error:", error)

            yield stream_event("error", message=str(error))

    return StreamingResponse(fallback_generator(), media_type="application/x-ndjson")


# =========================================================
# STREAMING /chat
# =========================================================


@app.post("/chat")
def chat(request: ChatRequest):

    process_memory(request.message)

    history = request.history or []

    prompt = build_ai_context(
        request.message, request.vision or {}, request.file_context, history
    )

    def generate_response():

        try:

            yield stream_event("start", route="ai", intent="chat")

            response = model_stream_chat(
                GENERAL_MODEL,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a personal AI assistant. "
                            "Use conversation history to understand "
                            "follow-up questions. Use memory for personal "
                            "facts. Use normal knowledge for factual "
                            "questions. Do not invent facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            for chunk in response:

                try:

                    content = chunk["message"]["content"]

                except Exception:

                    content = ""

                if content:

                    yield stream_event("chunk", content=content)

            yield stream_event("done")

        except Exception as error:

            yield stream_event("error", message=str(error))

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


# =========================================================
# SPEECH TO TEXT
# =========================================================


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:

        temp_file.write(await file.read())

        temp_path = temp_file.name

    try:

        text = transcribe_audio(temp_path)

        return {"text": text}

    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)


# =========================================================
# TEXT TO SPEECH
# =========================================================


@app.post("/speak")
def text_to_speech(request: SpeakRequest):

    audio_file = speak(request.text)

    return FileResponse(
        path=str(audio_file), media_type="audio/wav", filename="output.wav"
    )


# =========================================================
# CHAT + VOICE
# =========================================================


@app.post("/chat-with-voice")
def chat_with_voice(request: ChatRequest):

    process_memory(request.message)

    ai_text = generate_ai_response(
        request.message,
        request.vision or {},
        request.file_context,
        request.history or [],
    )

    audio_file = speak(ai_text)

    return FileResponse(
        path=str(audio_file), media_type="audio/wav", filename="response.wav"
    )


# =========================================================
# OBJECT DETECTION
# =========================================================


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = DEFAULT_CONFIDENCE):

    confidence = max(0.05, min(confidence, 0.95))

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:

        return {"success": False, "error": "Could not decode image"}

    vision_model, _ = get_vision_models()\n\n    results = vision_model.predict(
        source=frame,
        conf=confidence,
        imgsz=YOLO_IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )

    detections = []

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            confidence_value = float(box.conf[0].cpu().numpy())

            class_id = int(box.cls[0].cpu().numpy())

            class_name = vision_model.names[class_id]

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence_value * 100, 1),
                    "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                }
            )

    object_counts = {}

    for detection in detections:

        name = detection["class"]

        object_counts[name] = object_counts.get(name, 0) + 1

    height, width = frame.shape[:2]

    return {
        "success": True,
        "width": width,
        "height": height,
        "detections": detections,
        "object_count": len(detections),
        "objects": object_counts,
        "confidence_threshold": confidence,
    }


# =========================================================
# FACE DETECTION
# =========================================================


@app.post("/faces")
async def detect_faces(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:

        return {"success": False, "error": "Could not decode image"}

    _, face_app = get_vision_models()\n\n    faces = face_app.get(frame)

    detected_faces = []

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        current_embedding = normalize_embedding(face.embedding)

        name, confidence = recognize_embedding(current_embedding)

        detected_faces.append(
            {
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "name": name,
                "confidence": confidence,
            }
        )

    return {
        "success": True,
        "faces": detected_faces,
        "face_count": len(detected_faces),
        "database_people": list(FACE_DATABASE.keys()),
    }


# =========================================================
# RECOGNIZE EMBEDDING
# =========================================================


def recognize_embedding(current_embedding):

    current_embedding = normalize_embedding(current_embedding)

    if current_embedding is None:

        return ("Unknown", 0.0)

    if not FACE_DATABASE:

        return ("Unknown", 0.0)

    best_name = "Unknown"

    best_similarity = -1.0

    for name, stored_embedding in FACE_DATABASE.items():

        stored_embedding = normalize_embedding(stored_embedding)

        if stored_embedding is None:

            continue

        similarity = float(np.dot(current_embedding, stored_embedding))

        if similarity > best_similarity:

            best_similarity = similarity

            best_name = name

    if best_similarity >= FACE_MATCH_THRESHOLD:

        recognized_name = best_name

    else:

        recognized_name = "Unknown"

    confidence = round(max(0.0, best_similarity) * 100, 1)

    return (recognized_name, confidence)


# =========================================================
# FACE ENROLLMENT
# =========================================================


@app.post("/enroll")
async def enroll_face(name: str, file: UploadFile = File(...)):

    name = name.strip()

    if not name:

        return {"success": False, "error": "Name cannot be empty"}

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:

        return {"success": False, "error": "Could not decode image"}

    faces = face_app.get(frame)

    if len(faces) == 0:

        return {"success": False, "error": "No face detected"}

    if len(faces) > 1:

        return {
            "success": False,
            "error": "Multiple faces detected. " "Please use an image with one face.",
        }

    embedding = normalize_embedding(faces[0].embedding)

    if embedding is None:

        return {"success": False, "error": "Could not create face embedding"}

    FACE_DATABASE[name] = embedding

    save_face_database()

    return {
        "success": True,
        "message": f"Face enrolled successfully for {name}",
        "name": name,
        "total_people": len(FACE_DATABASE),
    }


# =========================================================
# FACE RECOGNITION
# =========================================================


@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:

        return {"success": False, "error": "Could not decode image"}

    faces = face_app.get(frame)

    results = []

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        current_embedding = normalize_embedding(face.embedding)

        recognized_name, confidence = recognize_embedding(current_embedding)

        results.append(
            {
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "name": recognized_name,
                "confidence": confidence,
            }
        )

    return {
        "success": True,
        "faces": results,
        "face_count": len(results),
        "database_people": list(FACE_DATABASE.keys()),
    }


# =========================================================
# FACE DATABASE STATUS
# =========================================================


@app.get("/faces/database")
def face_database():

    return {
        "success": True,
        "people": list(FACE_DATABASE.keys()),
        "count": len(FACE_DATABASE),
    }


# =========================================================
# CALENDAR API
# =========================================================


@app.get("/calendar")
def calendar_status():

    events = _load_calendar_events()

    return {
        "success": True,
        "events": events,
        "count": len(events),
    }


@app.get("/calendar/today")
def calendar_today():

    return {
        "success": True,
        "events": get_today_calendar_events(),
    }


@app.get("/calendar/tomorrow")
def calendar_tomorrow():

    return {
        "success": True,
        "events": get_tomorrow_calendar_events(),
    }


@app.get("/calendar/upcoming")
def calendar_upcoming(limit: int = 10):

    return {
        "success": True,
        "events": get_upcoming_calendar_events(limit),
    }


@app.post("/calendar")
def calendar_add(request: MemoryRequest):

    event_text = str(request.memory).strip()

    if not event_text:

        return {"success": False, "error": "Event cannot be empty."}

    try:

        event = add_calendar_event(event_text)

        return {
            "success": True,
            "message": (
                f"Added '{event['title']}' for "
                f"{_calendar_datetime_text(event['datetime'])}."
            ),
            "event": event,
        }

    except Exception as error:

        return {"success": False, "error": str(error)}


@app.delete("/calendar/{event_id}")
def calendar_delete(event_id: int):

    deleted = delete_calendar_event(event_id)

    if not deleted:

        return {"success": False, "error": "Event not found."}

    return {"success": True, "message": "Event deleted successfully."}


# =========================================================
# MEMORY API
# =========================================================


@app.get("/memory")
def memory_status():

    return {
        "success": True,
        "count": get_memory_count(),
        "memories": get_all_memories(),
    }


@app.post("/memory")
def create_memory(request: MemoryRequest):

    memory_text = str(request.memory).strip()

    category = str(request.category).strip() or "general"

    if not memory_text:

        return {"success": False, "error": "Memory cannot be empty."}

    from memory.memory import save_or_update_memory

    result = save_or_update_memory(memory_text, category)

    return {
        "success": True,
        "message": (
            "Memory saved successfully."
            if result["action"] == "created"
            else "Memory updated successfully."
        ),
        "action": result["action"],
        "id": result["id"],
        "memory": memory_text,
        "category": category,
    }


@app.get("/memory/search")
def memory_search(query: str, limit: int = 10):

    return {
        "success": True,
        "query": query,
        "results": search_memories(query, limit),
    }


@app.get("/memory/relevant")
def relevant_memory_search(query: str, limit: int = 5):

    return {
        "success": True,
        "query": query,
        "results": get_relevant_memories(query, limit),
    }


@app.get("/memory/recent")
def recent_memories(limit: int = 10):

    return {
        "success": True,
        "memories": get_recent_memories(limit),
    }


@app.delete("/memory/{memory_id}")
def remove_memory(memory_id: int):

    deleted = delete_memory(memory_id)

    if not deleted:

        return {"success": False, "error": "Memory not found."}

    return {"success": True, "message": "Memory deleted successfully."}


@app.delete("/memory")
def clear_memories():

    delete_all_memories()

    return {"success": True, "message": "All memories deleted."}


# =========================================================
# FILE READER
# =========================================================


@app.post("/file/read")
async def read_uploaded_file(file: UploadFile = File(...)):

    filename = file.filename or "uploaded_file"

    extension = os.path.splitext(filename)[1].lower()

    if extension not in (SUPPORTED_FILE_EXTENSIONS):

        return {
            "success": False,
            "error": "Unsupported file type. " "Supported: TXT, PDF, DOCX",
        }

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:

            temp_file.write(await file.read())

            temp_path = temp_file.name

        extracted_text = read_file(temp_path)

        if not extracted_text.strip():

            return {
                "success": False,
                "error": "No readable text was found " "inside the file.",
            }

        info = get_file_info(temp_path)

        return {
            "success": True,
            "filename": filename,
            "extension": extension,
            "characters": len(extracted_text),
            "text": extracted_text,
            "file_info": {"size_bytes": info["size_bytes"]},
        }

    except Exception as error:

        return {"success": False, "error": str(error)}

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# =========================================================
# FILE + OLLAMA
# =========================================================


@app.post("/file/ask")
async def ask_about_file(question: str, file: UploadFile = File(...)):

    filename = file.filename or "uploaded_file"

    extension = os.path.splitext(filename)[1].lower()

    if extension not in (SUPPORTED_FILE_EXTENSIONS):

        return {
            "success": False,
            "error": "Unsupported file type. " "Supported: TXT, PDF, DOCX",
        }

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:

            temp_file.write(await file.read())

            temp_path = temp_file.name

        extracted_text = read_file(temp_path)

        if not extracted_text.strip():

            return {
                "success": False,
                "error": "No readable text was found " "inside the file.",
            }

        question = question.strip()

        if not question:

            return {"success": False, "error": "Question cannot be empty."}

        if len(extracted_text) > MAX_FILE_CONTEXT:

            extracted_text = (
                extracted_text[:MAX_FILE_CONTEXT] + "\n\n"
                "[File content truncated because it is too large.]"
            )

        prompt = f"""
You are a helpful AI assistant.

The user has uploaded a file.

Filename:
{filename}

Below is the extracted content from the file:

---------------- FILE CONTENT ----------------

{extracted_text}

-------------- END FILE CONTENT --------------

Answer the user's question using the file content above.

User question:
{question}

Instructions:
- Use the uploaded file as the primary source.
- Do not invent information that is not present in the file.
- If the answer cannot be found in the file, clearly say so.
- Give a concise and useful answer.
"""

        response = model_chat(
            GENERAL_MODEL,
            [{"role": "user", "content": prompt}],
        )

        ai_response = response["message"]["content"]

        return {
            "success": True,
            "filename": filename,
            "question": question,
            "response": ai_response,
            "characters": len(extracted_text),
        }

    except Exception as error:

        return {"success": False, "error": str(error)}

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# =========================================================
# ROUTER TEST
# =========================================================


@app.post("/route")
def route_only(request: ChatRequest):

    history = request.history or []

    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    if is_weather_question(request.message) or is_weather_followup(
        request.message, history
    ):

        result = {"route": "tool", "intent": "weather"}

    # -----------------------------------------------------
    # WEB SEARCH
    # -----------------------------------------------------

    elif (
        (routing := route_input(request.message)).get("route") == "tool"
        and routing.get("intent") == "web_search"
    ):

        result = routing

    # -----------------------------------------------------
    # CALENDAR
    # -----------------------------------------------------

    elif is_calendar_question(request.message) or is_calendar_followup(
        request.message, history
    ):

        result = {"route": "tool", "intent": "calendar"}

    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    else:

        try:

            action_response = handle_action(request.message)

        except Exception:

            action_response = None

        if action_response is not None:

            result = {
                "route": "action",
                "intent": action_response.get("action", "command"),
            }

        else:

            result = route_input(request.message)

    return {"success": True, "input": request.message, "route": result}


# =========================================================
# SERVER START
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    #
