# =========================================================
# PHASE 19 — INTELLIGENT AI ROUTER
# =========================================================

import re

from model_manager import (
    FAST_MODEL,
    GENERAL_MODEL,
    COMPLEX_MODEL,
    CODING_MODEL,
)


# ---------------------------------------------------------
# ROUTING HELPERS
# ---------------------------------------------------------

def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


CALCULATOR_PATTERNS = (
    r"^\s*[-+]?\d+(?:\.\d+)?\s*[+\-*/%]\s*[-+]?\d+(?:\.\d+)?\s*$",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\b\d+(?:\.\d+)?\s*%\s*of\s*\d+(?:\.\d+)?\b",
    r"\b(?:solve|calculate|compute|find)\s+\d+.*[+\-*/%].*\d+",
    r"\bhow\s+much\s+is\s+\d+.*[+\-*/%].*\d+",
)

WEB_SEARCH_PATTERNS = (
    r"^search(?:\s+for|\s+the\s+web|\s+online)?\s+",
    r"^look\s+up\s+",
    r"\b(?:latest|current|recent|today|tonight)\b",
    r"\bnews\b",
    r"\bon\s+the\s+(?:internet|web)\b",
    r"\bonline\b",
    r"\baccording\s+to\s+(?:the\s+)?(?:internet|web)\b",
)

OPEN_APP_PREFIXES = ("open ", "launch ", "start ", "run ")
CLOSE_APP_PREFIXES = ("close ", "stop ", "exit ", "quit ")

FACE_PATTERNS = (
    "who am i",
    "recognize me",
    "recognise me",
    "identify me",
    "who is this",
    "identify this person",
    "recognize this person",
    "recognise this person",
)

OBJECT_PATTERNS = (
    "what am i looking at",
    "what is in front of me",
    "what do you see",
    "what can you see",
    "identify the objects",
    "detect objects",
    "what objects are there",
    "what is around me",
)

CODING_PATTERNS = (
    r"\bdebug(?:ging)?\b",
    r"\bfix\s+(?:this|the|my)\s+code\b",
    r"\b(?:code|coding|program|programming)\b",
    r"\b(?:error|exception|bug|syntax|compile|compiler)\b",
    r"\b(?:function|class|variable|array|pointer|recursion)\b",
    r"\b(?:python|c programming|language c|c\+\+|cpp|java|javascript|react|html|css|fastapi|api|sql)\b",
)

COMPLEX_PATTERNS = (
    r"\bin\s+detail\b",
    r"\bdetailed\b",
    r"\bexplain\s+(?:this\s+)?in\s+detail\b",
    r"\bexplain\s+deeply\b",
    r"\banaly[sz]e\b",
    r"\bin[-\s]depth\b",
    r"\bdeeply\b",
    r"\bstep[-\s]by[-\s]step\b",
    r"\b(?:prove|derive|reason|reasoning)\b",
    r"\b(?:compare|comparison)\b",
    r"\bpros\s+and\s+cons\b",
    r"\btrade[-\s]off\b",
    r"\bcomplex\b",
    r"\bsolve\s+this\s+problem\b",
)

# Strong signals justify the larger reasoning model. A weak signal such
# as the single word "complex" should not trigger a large-model call.
COMPLEX_STRONG_PATTERNS = (
    r"\bin\s+detail\b",
    r"\bdetailed\b",
    r"\bexplain\s+(?:this\s+)?in\s+detail\b",
    r"\bexplain\s+deeply\b",
    r"\banaly[sz]e\b",
    r"\bin[-\s]depth\b",
    r"\bdeeply\b",
    r"\bstep[-\s]by[-\s]step\b",
    r"\b(?:prove|derive|reason|reasoning)\b",
    r"\b(?:compare|comparison)\b",
    r"\bpros\s+and\s+cons\b",
    r"\btrade[-\s]off\b",
    r"\bsolve\s+this\s+problem\b",
)

FAST_PATTERNS = (
    r"^hi[!.]?$",
    r"^hello[!.]?$",
    r"^hey[!.]?$",
    r"^thanks[!.]?$",
    r"^thank\s+you[!.]?$",
    r"^good\s+(?:morning|afternoon|evening)[!.]?$",
    r"^who\s+are\s+you\??$",
    r"^what\s+can\s+you\s+do\??$",
)


def _route(route: str, intent: str, reason: str):
    return {
        "route": route,
        "intent": intent,
        "reason": reason,
    }


# ---------------------------------------------------------
# PHASE 19.1 — DETERMINISTIC ROUTING
# ---------------------------------------------------------
# Priority matters. Specific tool/action requests are checked
# before generic AI classification so that a message such as
# "calculate 20 + 30" never reaches an LLM.
# ---------------------------------------------------------

def route_input(user_input: str):
    text = str(user_input or "").lower().strip()

    if not text:
        return _route("ai", "chat", "empty_input")

    # 1. Exact/local calculator requests
    if _matches(text, CALCULATOR_PATTERNS):
        return _route("command", "calculator", "calculator_pattern")

    # 2. Explicit/current web requests
    if _matches(text, WEB_SEARCH_PATTERNS):
        return _route("tool", "web_search", "web_search_pattern")

    # 3. Desktop application commands
    if text.startswith(OPEN_APP_PREFIXES):
        return _route("command", "open_app", "open_app_prefix")

    if text.startswith(CLOSE_APP_PREFIXES):
        return _route("command", "close_app", "close_app_prefix")

    # 4. Vision requests
    if any(phrase in text for phrase in FACE_PATTERNS):
        return _route("vision", "face_recognition", "face_recognition_pattern")

    if any(phrase in text for phrase in OBJECT_PATTERNS):
        return _route("vision", "object_detection", "object_detection_pattern")

    # 5. Classify AI requests before model selection.
    intent = detect_ai_intent(text)

    return _route("ai", intent, f"intent_{intent}")


# ---------------------------------------------------------
# PHASE 19.2 — INTENT DETECTION
# ---------------------------------------------------------
# Intent detection happens after deterministic tool routing.
# This keeps tool/action requests out of the LLM path while
# giving normal AI requests a more precise intent before model
# selection.
# ---------------------------------------------------------

SIMPLE_QUESTION_PATTERNS = (
    r"^what\s+is\b",
    r"^what\s+are\b",
    r"^what\s+does\b",
    r"^define\b",
    r"^definition\s+of\b",
    r"^meaning\s+of\b",
    r"^who\s+is\b",
    r"^where\s+is\b",
    r"^when\s+is\b",
    r"^how\s+many\b",
    r"^how\s+much\b",
    r"^can\s+you\s+(?:tell|explain)\s+me\s+",
)

QUESTION_START_PATTERNS = (
    r"^(?:what|who|where|when|which|how|why|can|could|would|should|is|are|do|does|did)\b",
)


# =========================================================
# PHASE 19.4 — CONFIDENCE & FALLBACK ROUTING
# =========================================================
# Keep intent detection deterministic, but attach a confidence
# estimate so uncertain classifications can safely fall back
# to the general model instead of forcing a specialized model.
# =========================================================

INTENT_CONFIDENCE = {
    "greeting": 0.99,
    "simple_question": 0.95,
    "coding": 0.92,
    "complex_reasoning": 0.90,
    "normal_question": 0.80,
    "chat": 0.60,
}

FALLBACK_CONFIDENCE_THRESHOLD = 0.70


def detect_ai_intent_with_confidence(text: str) -> tuple[str, float]:
    """Classify an AI request and return (intent, confidence)."""

    text = str(text or "").lower().strip()

    if not text:
        return "chat", 0.0

    # More specific intents must win over broad question patterns.
    if _matches(text, CODING_PATTERNS):
        return "coding", INTENT_CONFIDENCE["coding"]

    if _matches(text, COMPLEX_PATTERNS):
        return "complex_reasoning", INTENT_CONFIDENCE["complex_reasoning"]

    if _matches(text, FAST_PATTERNS):
        return "greeting", INTENT_CONFIDENCE["greeting"]

    if _matches(text, SIMPLE_QUESTION_PATTERNS) and len(text) <= 100:
        return "simple_question", INTENT_CONFIDENCE["simple_question"]

    if "?" in text or _matches(text, QUESTION_START_PATTERNS):
        return "normal_question", INTENT_CONFIDENCE["normal_question"]

    return "chat", INTENT_CONFIDENCE["chat"]


def detect_ai_intent(text: str) -> str:
    """Compatibility wrapper that returns only the detected intent."""

    intent, _ = detect_ai_intent_with_confidence(text)
    return intent


# =========================================================
# PHASE 19.6 — MULTI-INTENT DETECTION
# =========================================================
# Detect multiple AI intents in one request without changing the
# existing deterministic tool/action priority. This lets model
# selection choose a suitable model for compound AI questions.
# =========================================================

AI_INTENT_PRIORITY = (
    "coding",
    "complex_reasoning",
    "normal_question",
    "simple_question",
    "greeting",
    "chat",
)


def detect_ai_intents(text: str) -> tuple[str, ...]:
    """Return all meaningful AI intents detected in a request."""

    text = str(text or "").lower().strip()

    if not text:
        return ("chat",)

    intents = []

    if _matches(text, CODING_PATTERNS):
        intents.append("coding")

    if _matches(text, COMPLEX_PATTERNS):
        intents.append("complex_reasoning")

    if _matches(text, FAST_PATTERNS):
        intents.append("greeting")

    if _matches(text, SIMPLE_QUESTION_PATTERNS) and len(text) <= 100:
        intents.append("simple_question")

    if "?" in text or _matches(text, QUESTION_START_PATTERNS):
        intents.append("normal_question")

    if not intents:
        return ("chat",)

    # Remove duplicate/overlapping classifications while keeping the
    # strongest intent first.
    return tuple(
        intent
        for intent in AI_INTENT_PRIORITY
        if intent in intents
    )


def is_multi_intent(text: str) -> bool:
    """Return True when a request contains two or more AI intents."""

    return len(detect_ai_intents(text)) > 1


# =========================================================
# PHASE 19.3 — INTELLIGENT MODEL SELECTION
# =========================================================
# Select the smallest suitable model for the detected intent.
# The goal is to preserve response quality while avoiding
# unnecessary calls to the larger models.
# =========================================================

INTENT_MODEL_MAP = {
    "greeting": FAST_MODEL,
    "simple_question": FAST_MODEL,
    "normal_question": GENERAL_MODEL,
    "chat": GENERAL_MODEL,
    "complex_reasoning": COMPLEX_MODEL,
    "coding": CODING_MODEL,
}


def select_model(
    user_input: str,
    route: str = "ai",
    intent: str = "chat",
    vision=None,
    file_context: str | None = None,
):
    text = str(user_input or "").lower().strip()

    if route != "ai":
        return None

    # Older callers may still pass the default "chat" intent.
    # Re-detect it so Phase 19.2 directly feeds Phase 19.3.
    confidence = INTENT_CONFIDENCE.get(intent, 0.0)

    if intent == "chat":
        intents = detect_ai_intents(text)
        intent = intents[0]
        confidence = INTENT_CONFIDENCE.get(intent, 0.0)

        # Compound AI requests should be handled by the strongest
        # required model rather than the smallest matching model.
        if len(intents) > 1:
            if "coding" in intents:
                return CODING_MODEL
            if "complex_reasoning" in intents:
                return COMPLEX_MODEL
            if "normal_question" in intents:
                return GENERAL_MODEL
            if "simple_question" in intents:
                return FAST_MODEL

    # Protect model selection from invalid/unknown intent values.
    if intent not in INTENT_MODEL_MAP:
        intent, confidence = detect_ai_intent_with_confidence(text)

    # Low-confidence classifications use the safe general model.
    # This prevents uncertain requests from being sent to a
    # specialized small model.
    if confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        return GENERAL_MODEL

    # Explicit coding intent always uses the coding-specialized model.
    # This also takes priority when a code-related file is attached.
    if intent == "coding":
        return CODING_MODEL

    # Explicit complex reasoning gets the larger reasoning model.
    # File context does not automatically upgrade it further.
    if intent == "complex_reasoning":
        # Phase 19.5: avoid large-model calls for weak, short matches.
        # Strong reasoning requests still use the complex model.
        if _matches(text, COMPLEX_STRONG_PATTERNS):
            return COMPLEX_MODEL
        if len(text) >= 120:
            return COMPLEX_MODEL
        return GENERAL_MODEL

    # A file needs more context capacity than the fast model, but
    # ordinary file questions do not automatically need the 4B model.
    if file_context:
        return GENERAL_MODEL

    # Use the centralized intent map for the remaining AI requests.
    selected_model = INTENT_MODEL_MAP.get(intent, GENERAL_MODEL)

    # Safety fallback for short conversational messages that were
    # not recognized by the classifier.
    if selected_model == GENERAL_MODEL and len(text) <= 30:
        if _matches(text, FAST_PATTERNS):
            return FAST_MODEL

    return selected_model


def route_request(user_input: str):
    return route_input(user_input)
