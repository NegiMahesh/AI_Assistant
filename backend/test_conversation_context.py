import unittest

from backend.conversation_context import (
    build_reference_context,
    compact_history,
    extract_topic,
    find_recent_topic,
    is_follow_up,
)


class TestPhase21ConversationContext(unittest.TestCase):

    def test_extract_python_topic(self):
        self.assertEqual(
            extract_topic("What is Python?"),
            "Python",
        )

    def test_extract_virtual_memory_topic(self):
        self.assertEqual(
            extract_topic("How does virtual memory work?"),
            "virtual memory",
        )

    def test_detect_follow_up_reference(self):
        self.assertTrue(is_follow_up("Who created it?"))
        self.assertTrue(is_follow_up("Why is that important?"))

    def test_new_topic_is_not_forced_into_reference(self):
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        self.assertEqual(
            build_reference_context("What is C++?", history),
            "",
        )

    def test_python_reference_context(self):
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
        ]

        result = build_reference_context(
            "Who created it?",
            history,
        )

        self.assertIn("Python", result)
        self.assertIn("What is Python?", result)

    def test_recent_topic_skips_short_acknowledgement(self):
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "okay"},
        ]

        topic, source = find_recent_topic(history)

        self.assertEqual(topic, "Python")
        self.assertEqual(source, "What is Python?")

    def test_history_message_limit(self):
        history = [
            {"role": "user", "content": f"message {index}"}
            for index in range(20)
        ]

        result = compact_history(
            history,
            max_messages=8,
        )

        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]["content"], "message 12")
        self.assertEqual(result[-1]["content"], "message 19")

    def test_individual_message_is_trimmed(self):
        long_text = "x" * 5000

        result = compact_history(
            [{"role": "user", "content": long_text}],
            max_messages=8,
            max_message_chars=1200,
        )

        self.assertLessEqual(
            len(result[0]["content"]),
            1200,
        )
        self.assertIn("[message truncated]", result[0]["content"])

    def test_total_history_is_bounded(self):
        history = [
            {
                "role": "user",
                "content": "x" * 1000,
            }
            for _ in range(12)
        ]

        result = compact_history(
            history,
            max_messages=12,
            max_message_chars=1200,
            max_history_chars=3000,
        )

        total_chars = sum(len(item["content"]) for item in result)

        self.assertLessEqual(total_chars, 3000)


if __name__ == "__main__":
    unittest.main()
