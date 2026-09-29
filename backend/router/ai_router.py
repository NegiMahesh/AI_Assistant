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
    r"\bwhy\s+(?:does|is|are)\b",
    r"\bhow\s+does\b",
    r"\bcomplex\b",
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

    # 5. Everything else goes to model selection.
    return _route("ai", "chat", "default_ai_route")


# ---------------------------------------------------------
# PHASE 19.2 PREPARATION — MODEL SELECTION
# ---------------------------------------------------------

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

    # Coding takes priority over generic "how/why" patterns.
    # Example: "why is this Python code failing?"
    if _matches(text, CODING_PATTERNS):
        return CODING_MODEL

    # Explicitly demanding deeper reasoning uses the complex model.
    if _matches(text, COMPLEX_PATTERNS):
        return COMPLEX_MODEL

    # Very short conversational requests are cheap enough for the
    # smallest model.
    if len(text) <= 30 and _matches(text, FAST_PATTERNS):
        return FAST_MODEL

    # File context needs enough capacity to understand the supplied
    # document, but does not automatically require the 4B model.
    if file_context:
        return GENERAL_MODEL

    return GENERAL_MODEL


def route_request(user_input: str):
    return route_input(user_input)
