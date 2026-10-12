import tempfile
import unittest
from pathlib import Path

from backend.memory import memory as memory_store

detect_memory = memory_store.detect_memory


class TestPhase21AgeMemory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_file = memory_store.DATABASE_FILE
        memory_store.DATABASE_FILE = Path(self.temp_dir.name) / "test-memory.db"
        memory_store.initialize_database()

    def tearDown(self):
        memory_store.DATABASE_FILE = self.original_database_file
        self.temp_dir.cleanup()

    def test_age_with_years_old_is_detected(self):
        result = detect_memory("I am 20 years old.")
        self.assertEqual(result["category"], "age")
        self.assertEqual(result["memory"], "The user's age is 20 years old.")

    def test_age_without_suffix_is_detected(self):
        result = detect_memory("I'm 20")
        self.assertEqual(result["category"], "age")
        self.assertIn("20 years old", result["memory"])

    def test_age_is_detected_from_my_age_is(self):
        result = detect_memory("My age is 20.")
        self.assertEqual(result["category"], "age")
        self.assertIn("20 years old", result["memory"])

    def test_age_statement_with_currently_is_detected(self):
        result = detect_memory("I am currently 20 years old")
        self.assertEqual(result["category"], "age")
        self.assertIn("20 years old", result["memory"])

    def test_age_out_of_range_is_not_saved(self):
        self.assertIsNone(detect_memory("I am 150 years old"))

    def test_question_is_not_mistaken_for_a_memory_statement(self):
        self.assertIsNone(detect_memory("How old am I?"))

    def test_currently_phrased_age_is_saved_and_retrieved(self):
        result = memory_store.process_memory("I am currently 20 years old.")
        self.assertEqual(result["category"], "age")

        memories = memory_store.get_relevant_memories("How old am I?")
        self.assertTrue(any(
            item["category"] == "age" and "20 years old" in item["memory"]
            for item in memories
        ))

    def test_compound_question_retrieves_both_name_and_age(self):
        memory_store.process_memory("My name is Mahesh.")
        memory_store.process_memory("I am currently 20 years old.")

        memories = memory_store.get_relevant_memories(
            "Now tell me whats my name and how old am I",
            limit=5,
        )
        memory_texts = [item["memory"] for item in memories]

        self.assertTrue(any("name is Mahesh" in item for item in memory_texts))
        self.assertTrue(any("20 years old" in item for item in memory_texts))

    def test_new_age_updates_the_stored_age(self):
        memory_store.process_memory("I am 20 years old.")
        memory_store.process_memory("I am 21 years old.")

        ages = [
            item for item in memory_store.get_all_memories()
            if item["category"] == "age"
        ]
        self.assertEqual(len(ages), 1)
        self.assertIn("21 years old", ages[0]["memory"])


if __name__ == "__main__":
    unittest.main()
