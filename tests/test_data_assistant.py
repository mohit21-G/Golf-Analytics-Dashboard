import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from rag.data_assistant import answer_data_query, _time_based_greeting, _extract_date, _load_data


class TestDataAssistant(unittest.TestCase):
    def test_list_all_club_names(self):
        answer = answer_data_query("give all club names")
        self.assertIsNotNone(answer)
        self.assertTrue("| Course Name |" in answer or "<table" in answer)

    def test_data_for_specific_date(self):
        answer = answer_data_query("data for 2026-05-08")
        self.assertIsNotNone(answer)
        self.assertTrue("| Course Name |" in answer or "<table" in answer)
        self.assertTrue("| Date |" in answer or "<table" in answer)
        self.assertIn("2026-05-08", answer)

    def test_highest_occupancy(self):
        answer = answer_data_query("highest occupancy")
        self.assertIsNotNone(answer)
        self.assertTrue("| Course Name |" in answer or "<table" in answer)
        self.assertTrue("| Occupancy |" in answer or "<table" in answer)

    def test_spelling_and_synonym_variation(self):
        answer = answer_data_query("blu heron")
        self.assertIsNotNone(answer)
        self.assertIn("Blue Heron", answer)
        self.assertTrue("| Course Name |" in answer or "<table" in answer)

    def test_keyword_case_insensitive_table(self):
        answer = answer_data_query("SHOW PRICE FOR STONEGATE")
        self.assertIsNotNone(answer)
        self.assertIn("Stonegate", answer)
        self.assertTrue("| Price |" in answer or "<table" in answer)

    def test_small_talk_response(self):
        answer = answer_data_query("how are you")
        self.assertIsNotNone(answer)
        self.assertIn("I'm doing great!", answer)
        self.assertIn("How can I help you today?", answer)

    def test_user_fine_flow_response(self):
        answer = answer_data_query("I am good")
        self.assertIsNotNone(answer)
        self.assertIn("That's great to hear!", answer)
        self.assertIn("prices, occupancy", answer)

    def test_explicit_good_morning_response(self):
        answer = answer_data_query("good morning")
        self.assertIsNotNone(answer)
        self.assertIn("How can I help you?", answer)

    def test_time_based_greeting_morning(self):
        greeting = _time_based_greeting(datetime(2026, 5, 5, 9, 0, 0))
        self.assertEqual(greeting, "Good morning")

    def test_time_based_greeting_afternoon(self):
        greeting = _time_based_greeting(datetime(2026, 5, 5, 14, 0, 0))
        self.assertEqual(greeting, "Good afternoon")

    def test_time_based_greeting_evening(self):
        greeting = _time_based_greeting(datetime(2026, 5, 5, 20, 0, 0))
        self.assertEqual(greeting, "Good evening")

    def test_generic_hello_uses_simple_message(self):
        with patch("rag.data_assistant.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 5, 5, 19, 0, 0)
            answer = answer_data_query("hello")
        self.assertEqual(answer, "Hello! 😊 How can I help you?")

    def test_extract_date_ordinal_day_month(self):
        result = _extract_date("show me data for 7th July", available_dates={"2026-07-07", "2026-05-05"})
        self.assertEqual(result, "2026-07-07")

    def test_extract_date_month_day(self):
        result = _extract_date("price for July 7", available_dates={"2026-07-07", "2026-05-05"})
        self.assertEqual(result, "2026-07-07")

    def test_extract_date_numeric(self):
        result = _extract_date("availability on 07-07-2026")
        self.assertEqual(result, "2026-07-07")

    def test_no_data_message_for_exact_date_when_missing(self):
        answer = answer_data_query("show data for 07-07-2026")
        self.assertEqual(answer, "No data available for 2026-07-07")

    def test_fuzzy_phrase_day_month_without_year_uses_dataset_year(self):
        answer = answer_data_query("I want 7th july data")
        self.assertIsNotNone(answer)
        self.assertNotIn("2026-07-05", answer)
        if "No data available for 2026-07-07" not in answer:
            self.assertIn("<table", answer)
            self.assertIn("2026-07-07", answer)

    def test_fuzzy_phrase_month_day(self):
        answer = answer_data_query("show july 7 details")
        self.assertIsNotNone(answer)
        self.assertNotIn("2026-07-05", answer)
        if "No data available for 2026-07-07" not in answer:
            self.assertIn("<table", answer)
            self.assertIn("2026-07-07", answer)

    def test_fuzzy_phrase_day_month(self):
        answer = answer_data_query("give me data for 7 july")
        self.assertIsNotNone(answer)
        self.assertNotIn("2026-07-05", answer)
        if "No data available for 2026-07-07" not in answer:
            self.assertIn("<table", answer)
            self.assertIn("2026-07-07", answer)

    def test_valid_dataset_date_returns_data(self):
        store = _load_data()
        dates = sorted(set(store.market["tee_date_iso"].dropna().astype(str)) | set(store.availability["tee_date_iso"].dropna().astype(str)))
        self.assertTrue(len(dates) > 0)
        sample_date = dates[0]
        answer = answer_data_query(f"show data for {sample_date}")
        self.assertIsNotNone(answer)
        self.assertIn(sample_date, answer)
        self.assertNotEqual(answer, "No data available for this date")

    def test_exact_date_query_returns_all_matching_rows_without_limit(self):
        store = _load_data()
        combined = pd.concat(
            [
                store.market[["tee_date_iso"]].assign(source="market"),
                store.availability[["tee_date_iso"]].assign(source="availability"),
            ],
            ignore_index=True,
        )
        counts = combined["tee_date_iso"].value_counts()
        self.assertTrue(len(counts) > 0)
        date_with_most_rows = str(counts.index[0])
        expected_rows = int(counts.iloc[0])

        answer = answer_data_query(f"show data for {date_with_most_rows}")
        self.assertIsNotNone(answer)
        self.assertIn("<table", answer)
        # Date queries now render market and availability as separate tables.
        # Total <tr> count = all data rows + one header row per table rendered.
        # We verify all data rows are present: <tr> count >= expected_rows.
        self.assertGreaterEqual(answer.count("<tr>"), expected_rows)
        self.assertNotIn("Showing ", answer)

    def test_positive_mood_variations(self):
        positives = ["I am fine", "doing well", "not bad", "okay", "I'm great"]
        for p in positives:
            with self.subTest(p=p):
                answer = answer_data_query(p)
                self.assertIsNotNone(answer)
                self.assertIn("That's great to hear!", answer)

    def test_negative_mood_variations(self):
        negatives = ["not good", "I'm not well", "feeling bad", "sad"]
        for p in negatives:
            with self.subTest(p=p):
                answer = answer_data_query(p)
                self.assertIsNotNone(answer)
                self.assertIn("I'm sorry to hear that", answer)


if __name__ == "__main__":
    unittest.main()
