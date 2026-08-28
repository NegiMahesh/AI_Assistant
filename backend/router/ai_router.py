# =========================================================
# AI ROUTER
# =========================================================

import re


# =========================================================
# ROUTE INPUT
# =========================================================

def route_input(user_input: str):

    text = user_input.lower().strip()

    # =====================================================
    # CALCULATOR
    # =====================================================

    calculator_patterns = [
        r"\d+\s*[\+\-\*/]\s*\d+",
        r"calculate",
        r"what is .*%",
        r"\d+\s*%\s*of\s*\d+",
        r"solve .*[+\-\*/]",
        r"how much is .*[+\-\*/]"
    ]

    for pattern in calculator_patterns:

        if re.search(pattern, text):

            return {
                "route": "command",
                "intent": "calculator"
            }

    # =====================================================
    # WEB SEARCH
    # =====================================================

    web_search_patterns = [

        # Explicit search commands
        r"^search\s+",
        r"^search\s+for\s+",
        r"^search\s+the\s+web\s+",
        r"^search\s+online\s+",
        r"^look\s+up\s+",

        # Current/latest information
        r"\blatest\b",
        r"\bcurrent\b",
        r"\btoday's\b",
        r"\btoday\b",
        r"\brecent\b",
        r"\bnews\b",

        # Internet-related requests
        r"\bon\s+the\s+internet\b",
        r"\bonline\b",
        r"\baccording\s+to\s+the\s+internet\b",
    ]

    for pattern in web_search_patterns:

        if re.search(pattern, text):

            return {
                "route": "tool",
                "intent": "web_search"
            }

    # =====================================================
    # OPEN APPLICATION
    # =====================================================

    open_patterns = [
        "open ",
        "launch ",
        "start ",
        "run "
    ]

    for pattern in open_patterns:

        if text.startswith(pattern):

            return {
                "route": "command",
                "intent": "open_app"
            }

    # =====================================================
    # CLOSE APPLICATION
    # =====================================================

    close_patterns = [
        "close ",
        "stop ",
        "exit ",
        "quit "
    ]

    for pattern in close_patterns:

        if text.startswith(pattern):

            return {
                "route": "command",
                "intent": "close_app"
            }

    # =====================================================
    # FACE RECOGNITION
    # =====================================================

    face_phrases = [

        "who am i",
        "recognize me",
        "recognise me",
        "identify me",
        "who is this",
        "identify this person",
        "recognize this person",
        "recognise this person"

    ]

    for phrase in face_phrases:

        if phrase in text:

            return {
                "route": "vision",
                "intent": "face_recognition"
            }

    # =====================================================
    # OBJECT DETECTION
    # =====================================================

    vision_phrases = [

        "what am i looking at",
        "what is in front of me",
        "what do you see",
        "what can you see",
        "identify the objects",
        "detect objects",
        "what objects are there",
        "what is around me"

    ]

    for phrase in vision_phrases:

        if phrase in text:

            return {
                "route": "vision",
                "intent": "object_detection"
            }

    # =====================================================
    # DEFAULT → AI
    # =====================================================

    return {
        "route": "ai",
        "intent": "chat"
    }


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def route_request(user_input: str):

    return route_input(user_input)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_inputs = [

        "What is 25 * 40?",
        "Calculate 100 + 50",
        "What is 15% of 200?",

        "Search for latest Python version",
        "Search the web for AI news",
        "Look up Python 3.14",
        "What is the latest technology news?",
        "What is the current population of India?",

        "Open Chrome",
        "Launch calculator",

        "Close Chrome",

        "What is polymorphism?",
        "Explain machine learning",
        "Tell me a joke",

        "What am I looking at?",
        "What is in front of me?",

        "Who am I?",
        "Recognize me"
    ]

    print()
    print("=" * 60)
    print("AI ROUTER TEST")
    print("=" * 60)

    for user_input in test_inputs:

        result = route_input(
            user_input
        )

        print()
        print("INPUT :", user_input)
        print("ROUTE :", result["route"])
        print("INTENT:", result["intent"])

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)