"""
Phase 19.7 — Intelligent routing test suite.

Run from the project root:
    python -m unittest discover -s backend/router -p "test_ai_router.py" -v

The tests cover deterministic routes, AI intents, model selection,
confidence fallback, and multi-intent requests.
"""

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from router.ai_router import (  # noqa: E402
    CODING_MODEL,
    COMPLEX_MODEL,
    FAST_MODEL,
    GENERAL_MODEL,
    FALLBACK_CONFIDENCE_THRESHOLD,
    detect_ai_intent,
    detect_ai_intent_with_confidence,
    detect_ai_intents,
    is_multi_intent,
    route_input,
    select_model,
)


class TestPhase197Routing(unittest.TestCase):
    """Regression tests for the complete Phase 19 router."""

    def assert_route(self, text, route, intent):
        result = route_input(text)
        self.assertEqual(result["route"], route, text)
        self.assertEqual(result["intent"], intent, text)

    # -----------------------------------------------------
    # Deterministic routes
    # -----------------------------------------------------

    def test_calculator_route(self):
        self.assert_route("calculate 25 + 15", "command", "calculator")

    def test_web_search_route(self):
        self.assert_route("search for the latest Python news", "tool", "web_search")

    def test_open_app_route(self):
        self.assert_route("open chrome", "command", "open_app")

    def test_close_app_route(self):
        self.assert_route("close chrome", "command", "close_app")

    def test_face_route(self):
        self.assert_route("who am i", "vision", "face_recognition")

    def test_object_detection_route(self):
        self.assert_route("what objects are there", "vision", "object_detection")

    # -----------------------------------------------------
    # AI intent routes
    # -----------------------------------------------------

    def test_greeting_intent(self):
        self.assert_route("Hello", "ai", "greeting")

    def test_simple_question_intent(self):
        self.assert_route("What is RAM?", "ai", "simple_question")

    def test_normal_question_intent(self):
        self.assert_route("How does virtual memory work?", "ai", "normal_question")

    def test_complex_reasoning_intent(self):
        self.assert_route(
            "Explain RISC vs CISC in detail",
            "ai",
            "complex_reasoning",
        )

    def test_coding_intent(self):
        self.assert_route(
            "Why is my Python code giving an error?",
            "ai",
            "coding",
        )

    def test_ambiguous_chat_intent(self):
        self.assert_route("Tell me something interesting", "ai", "chat")

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    def test_confidence_values_are_in_expected_range(self):
        examples = {
            "Hello": "greeting",
            "What is RAM?": "simple_question",
            "How does virtual memory work?": "normal_question",
            "Explain RISC vs CISC in detail": "complex_reasoning",
            "Why is my Python code giving an error?": "coding",
            "Tell me something interesting": "chat",
        }

        for text, expected_intent in examples.items():
            intent, confidence = detect_ai_intent_with_confidence(text)
            self.assertEqual(intent, expected_intent, text)
            self.assertGreaterEqual(confidence, 0.0, text)
            self.assertLessEqual(confidence, 1.0, text)

    def test_low_confidence_chat_falls_back_to_general_model(self):
        intent, confidence = detect_ai_intent_with_confidence(
            "Tell me something interesting"
        )
        self.assertEqual(intent, "chat")
        self.assertLess(confidence, FALLBACK_CONFIDENCE_THRESHOLD)
        self.assertEqual(
            select_model(
                "Tell me something interesting",
                route="ai",
                intent="chat",
            ),
            GENERAL_MODEL,
        )

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------

    def test_fast_model_for_greeting(self):
        self.assertEqual(
            select_model("Hello", route="ai", intent="greeting"),
            FAST_MODEL,
        )

    def test_fast_model_for_simple_question(self):
        self.assertEqual(
            select_model("What is RAM?", route="ai", intent="simple_question"),
            FAST_MODEL,
        )

    def test_general_model_for_normal_question(self):
        self.assertEqual(
            select_model(
                "How does virtual memory work?",
                route="ai",
                intent="normal_question",
            ),
            GENERAL_MODEL,
        )

    def test_complex_model_for_strong_complex_request(self):
        self.assertEqual(
            select_model(
                "Explain RISC vs CISC in detail and compare their advantages",
                route="ai",
                intent="complex_reasoning",
            ),
            COMPLEX_MODEL,
        )

    def test_coding_model_for_coding_request(self):
        self.assertEqual(
            select_model(
                "Explain this Python error and tell me how to fix it",
                route="ai",
                intent="coding",
            ),
            CODING_MODEL,
        )

    def test_weak_complex_request_stays_on_general_model(self):
        self.assertEqual(
            select_model(
                "This is complex",
                route="ai",
                intent="complex_reasoning",
            ),
            GENERAL_MODEL,
        )

    # -----------------------------------------------------
    # Multi-intent detection
    # -----------------------------------------------------

    def test_multi_intent_question(self):
        text = "Hello, can you explain what a pointer is?"
        intents = detect_ai_intents(text)

        self.assertTrue(is_multi_intent(text))
        self.assertIn("greeting", intents)
        self.assertIn("normal_question", intents)

        # This is an ordinary technical question, not a request to
        # write/debug code, so it should avoid the coding model.
        self.assertEqual(
            select_model(text, route="ai", intent="chat"),
            GENERAL_MODEL,
        )

    def test_multi_intent_coding_request_uses_coding_model(self):
        text = "Can you explain this Python error and tell me how to fix it?"
        intents = detect_ai_intents(text)

        self.assertTrue(is_multi_intent(text))
        self.assertIn("coding", intents)
        self.assertIn("normal_question", intents)
        self.assertEqual(
            select_model(text, route="ai", intent="chat"),
            CODING_MODEL,
        )

    def test_multi_intent_complex_request_uses_complex_model(self):
        text = "Can you explain RISC vs CISC in detail and compare their advantages?"
        intents = detect_ai_intents(text)

        self.assertIn("complex_reasoning", intents)
        self.assertIn("normal_question", intents)
        self.assertEqual(
            select_model(text, route="ai", intent="chat"),
            COMPLEX_MODEL,
        )

    def test_non_ai_route_does_not_select_llm(self):
        self.assertIsNone(
            select_model(
                "calculate 25 + 15",
                route="command",
                intent="calculator",
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
