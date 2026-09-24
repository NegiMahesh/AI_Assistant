# =========================================================
# AI ROUTER
# =========================================================

import re


def route_input(user_input: str):
    text = user_input.lower().strip()

    calculator_patterns = [
        r"^\s*\d+(?:\.\d+)?\s*[\+\-\*/%]\s*\d+(?:\.\d+)?\s*$",
        r"\bcalculate\b",
        r"\bwhat\s+is\s+\d+.*[\+\-\*/%].*\d+",
        r"\b\d+\s*%\s*of\s*\d+\b",
        r"\bsolve\s+\d+.*[\+\-\*/%].*\d+",
        r"\bhow\s+much\s+is\s+\d+.*[\+\-\*/%].*\d+",
    ]
    for pattern in calculator_patterns:
        if re.search(pattern, text):
            return {"route": "command", "intent": "calculator"}

    web_search_patterns = [
        r"^search\s+",
        r"^search\s+for\s+",
        r"^search\s+the\s+web\s+",
        r"^search\s+online\s+",
        r"^look\s+up\s+",
        r"\blatest\b",
        r"\bcurrent\b",
        r"\btoday'?s\b",
        r"\btoday\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\bon\s+the\s+internet\b",
        r"\bonline\b",
        r"\baccording\s+to\s+the\s+internet\b",
    ]
    for pattern in web_search_patterns:
        if re.search(pattern, text):
            return {"route": "tool", "intent": "web_search"}

    for pattern in ["open ", "launch ", "start ", "run "]:
        if text.startswith(pattern):
            return {"route": "command", "intent": "open_app"}

    for pattern in ["close ", "stop ", "exit ", "quit "]:
        if text.startswith(pattern):
            return {"route": "command", "intent": "close_app"}

    for phrase in [
        "who am i", "recognize me", "recognise me", "identify me",
        "who is this", "identify this person", "recognize this person",
        "recognise this person",
    ]:
        if phrase in text:
            return {"route": "vision", "intent": "face_recognition"}

    for phrase in [
        "what am i looking at", "what is in front of me", "what do you see",
        "what can you see", "identify the objects", "detect objects",
        "what objects are there", "what is around me",
    ]:
        if phrase in text:
            return {"route": "vision", "intent": "object_detection"}

    return {"route": "ai", "intent": "chat"}


def select_model(user_input: str, route: str = "ai", intent: str = "chat", vision=None, file_context: str | None = None):
    text = user_input.lower().strip()

    if route != "ai":
        return None

    coding_patterns = [
        r"\bdebug\b", r"\bdebugging\b", r"\bfix\s+(this|the|my)\s+code\b",
        r"\bcode\b", r"\bcoding\b", r"\bprogram\b", r"\bprogramming\b",
        r"\berror\b", r"\bexception\b", r"\bbug\b", r"\bsyntax\b",
        r"\bcompile\b", r"\bcompiler\b", r"\bfunction\b", r"\bclass\b",
        r"\bvariable\b", r"\barray\b", r"\bpointer\b", r"\brecursion\b",
        r"\bpython\b", r"\bc programming\b", r"\blanguage c\b", r"\bc\+\+\b",
        r"\bcpp\b", r"\bjava\b", r"\bjavascript\b", r"\breact\b",
        r"\bhtml\b", r"\bcss\b", r"\bfastapi\b", r"\bapi\b", r"\bsql\b",
    ]
    if any(re.search(p, text) for p in coding_patterns):
        return "qwen2.5-coder:3b"

    complex_patterns = [
        r"\bin\s+detail\b", r"\bdetailed\b", r"\bdetailed\s+explanation\b",
        r"\bexplain\s+in\s+detail\b", r"\bexplain\s+this\s+in\s+detail\b",
        r"\bexplain\s+deeply\b", r"\banalyze\b", r"\banalyse\b", r"\bin[-\s]depth\b",
        r"\bdeeply\b", r"\bstep[-\s]by[-\s]step\b", r"\bprove\b", r"\bderive\b",
        r"\breason\b", r"\breasoning\b", r"\bcompare\b", r"\bcomparison\b",
        r"\bpros\s+and\s+cons\b", r"\btrade[-\s]off\b", r"\bwhy\s+does\b",
        r"\bwhy\s+is\b", r"\bhow\s+does\b", r"\bcomplex\b",
        r"\bsolve\s+this\s+problem\b",
    ]
    if any(re.search(p, text) for p in complex_patterns):
        return "qwen3:4b"

    simple_patterns = [
        r"^hi$", r"^hello$", r"^hey$", r"^thanks$", r"^thank\s+you$",
        r"^good\s+morning$", r"^good\s+afternoon$", r"^good\s+evening$",
        r"^who\s+are\s+you\??$", r"^what\s+can\s+you\s+do\??$",
    ]
    if len(text) <= 30 and any(re.search(p, text) for p in simple_patterns):
        return "qwen3:0.6b"

    if file_context:
        return "qwen3:1.7b"

    return "qwen3:1.7b"


def route_request(user_input: str):
    return route_input(user_input)
