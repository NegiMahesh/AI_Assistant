"""Regression tests for durable personal-memory facts."""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure imports work when tests are launched from the repository root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from memory import memory as memory_store  # noqa: E402


class PersistentMemoryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_file = memory_store.DATABASE_FILE
        memory_store.DATABASE_FILE = Path(self.temp_dir.name) / "test-memory.db"
        memory_store.initialize_database()

    def tearDown(self):
        memory_store.DATABASE_FILE = self.original_database_file
        self.temp_dir.cleanup()

    def test_detect_age_from_common_phrasings(self):
        cases = (
            "I am 20 years old.",
            "I'm 20.",
            "My age is 20",
        )
        for message in cases:
            with self.subTest(message=message):
                detected = memory_store.detect_memory(message)
                self.assertEqual(detected["memory"], "The user's age is 20 years old.")
                self.assertEqual(detected["category"], "identity")

    def test_age_is_not_detected_from_a_question(self):
        self.assertIsNone(memory_store.detect_memory("How old am I?"))

    def test_age_is_saved_and_updated_without_conflicting_duplicates(self):
        first = memory_store.process_memory("I am 20 years old.")
        second = memory_store.process_memory("I am 21 years old.")

        self.assertTrue(first["saved"])
        self.assertEqual(second["action"], "updated")
        age_memories = [
            item for item in memory_store.get_all_memories()
            if item["category"] == "identity"
            and item["memory"].startswith("The user's age is ")
        ]
        self.assertEqual(len(age_memories), 1)
        self.assertEqual(age_memories[0]["memory"], "The user's age is 21 years old.")

    def test_age_can_be_retrieved_by_a_natural_question(self):
        memory_store.process_memory("I am 20 years old.")
        results = memory_store.get_relevant_memories("How old am I?")
        self.assertTrue(any("20 years old" in item["memory"] for item in results))

    def test_memory_context_formats_records_not_query_characters(self):
        memory_store.process_memory("I am 20 years old.")
        context = memory_store.build_memory_context(memory_store.get_recent_memories(10))
        self.assertIn("The user's age is 20 years old.", context)
        self.assertNotIn("- H (category: general)", context)


if __name__ == "__main__":
    unittest.main()
