import unittest

from backend.memory.memory import detect_memory


class TestPhase21AgeMemory(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
