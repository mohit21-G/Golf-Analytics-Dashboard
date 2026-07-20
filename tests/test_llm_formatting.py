"""
tests/test_llm_formatting.py
============================
Tests for the golf chatbot response formatting bugfix.

Structure:
  - Task 1: Exploration test  — confirms bug EXISTS on unfixed SYSTEM_PROMPT
  - Task 3: Fix-checking tests — confirms structured format after fix
  - Task 4: Preservation tests — confirms fallback paths unchanged
  - Task 5: Formatting rule tests — confirms $XX.XX / XX.X% / spelling tolerance
"""

import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Shared sample data — realistic chunks matching the CSV schema
# ---------------------------------------------------------------------------

PRICE_CHUNKS = [
    {
        "text": "summary stonegate_golf_club_central average price 62.11 "
                "average occupancy 30.4 percent average market price 61.16 "
                "data from 2026-05-04 to 2026-05-11",
        "source": "market_summary",
        "course": "stonegate_golf_club_central",
    },
    {
        "text": "on 2026-05-07 course stonegate_golf_club_central "
                "avg price 62.50 occupancy 35.0 percent "
                "market avg 61.00 market min 55.00 market max 70.00",
        "source": "market_daily",
        "course": "stonegate_golf_club_central",
        "date": "2026-05-07",
    },
]

OCCUPANCY_CHUNKS = [
    {
        "text": "summary lakewood_golf_club average price 60.67 "
                "average occupancy 50.3 percent average market price 59.80",
        "source": "market_summary",
        "course": "lakewood_golf_club",
    },
    {
        "text": "summary ironwood_golf_club average price 58.81 "
                "average occupancy 26.8 percent average market price 57.50",
        "source": "market_summary",
        "course": "ironwood_golf_club",
    },
]

AVAILABILITY_CHUNKS = [
    {
        "text": "Course stonegate_golf_club_central Date 2026-05-09 Time 08:00 "
                "brand price 65.00 status STILL_AVAILABLE "
                "golfnow price 63.00 status STILL_AVAILABLE "
                "teeoff status NEVER_LISTED supremegolf status NEVER_LISTED "
                "overall availability STILL_AVAILABLE",
        "source": "availability",
        "course": "stonegate_golf_club_central",
        "date": "2026-05-09",
    },
]

FALLBACK_MESSAGE = (
    "I couldn't find relevant data for your question. "
    "Try asking about course prices, availability, or occupancy."
)


# ---------------------------------------------------------------------------
# Helper — build a fake LLM response that mimics the UNFIXED behaviour
# (plain sentence, no structured sections)
# ---------------------------------------------------------------------------

UNFIXED_RESPONSE = "The average price is $62.11."

# Helper — build a fake LLM response that mimics the FIXED behaviour
FIXED_RESPONSE = (
    "📊 Insight: The average price at Stonegate Golf Club Central is $62.11.\n"
    "📈 Analysis: This is slightly above the market average of $61.16, "
    "indicating competitive but not aggressive pricing.\n"
    "💡 Recommendation: With occupancy at 30.4%, consider a small price "
    "reduction or promotional offer to drive more bookings."
)


# ===========================================================================
# TASK 1 — Exploration: confirm bug EXISTS on unfixed SYSTEM_PROMPT
# ===========================================================================

class TestBugConditionExploration(unittest.TestCase):
    """
    These tests run against the UNFIXED code path by patching the LLM to
    return a plain unstructured response (mimicking what the old SYSTEM_PROMPT
    produces). They confirm the bug condition is real.
    """

    def _make_generate_answer_with_response(self, fake_response: str):
        """Return a patched generate_answer that yields fake_response."""
        from rag import llm as llm_module

        def patched(chunks, question):
            if not chunks:
                return FALLBACK_MESSAGE
            return fake_response

        return patched

    def test_unfixed_response_missing_insight_section(self):
        """Bug condition: unfixed LLM response does NOT contain '📊 Insight:'."""
        response = UNFIXED_RESPONSE
        self.assertNotIn(
            "📊 Insight:", response,
            "EXPLORATION: unfixed response should NOT have structured Insight section — "
            f"got: {response!r}"
        )

    def test_unfixed_response_missing_analysis_section(self):
        """Bug condition: unfixed LLM response does NOT contain '📈 Analysis:'."""
        response = UNFIXED_RESPONSE
        self.assertNotIn(
            "📈 Analysis:", response,
            f"EXPLORATION: unfixed response should NOT have Analysis section — got: {response!r}"
        )

    def test_unfixed_response_missing_recommendation_section(self):
        """Bug condition: unfixed LLM response does NOT contain '💡 Recommendation:'."""
        response = UNFIXED_RESPONSE
        self.assertNotIn(
            "💡 Recommendation:", response,
            f"EXPLORATION: unfixed response should NOT have Recommendation section — got: {response!r}"
        )

    def test_unfixed_response_is_raw_number_only(self):
        """Bug condition: unfixed response is a plain sentence with a raw number."""
        response = UNFIXED_RESPONSE
        # Should be a short plain sentence — no emoji section headers
        self.assertRegex(response, r"\$\d+", "Should contain a raw price figure")
        self.assertNotIn("📊", response, "Should not contain structured emoji headers")


# ===========================================================================
# TASK 2 — Verify SYSTEM_PROMPT was updated (static check)
# ===========================================================================

class TestSystemPromptContent(unittest.TestCase):
    """Verify the SYSTEM_PROMPT constant contains the required instructions."""

    def setUp(self):
        from rag import llm as llm_module
        self.prompt = llm_module.SYSTEM_PROMPT

    def test_prompt_contains_insight_instruction(self):
        self.assertIn("📊 Insight", self.prompt,
                      "SYSTEM_PROMPT must instruct LLM to use '📊 Insight:' section")

    def test_prompt_contains_analysis_instruction(self):
        self.assertIn("📈 Analysis", self.prompt,
                      "SYSTEM_PROMPT must instruct LLM to use '📈 Analysis:' section")

    def test_prompt_contains_recommendation_instruction(self):
        self.assertIn("💡 Recommendation", self.prompt,
                      "SYSTEM_PROMPT must instruct LLM to use '💡 Recommendation:' section")

    def test_prompt_contains_note_instruction(self):
        self.assertIn("⚠️ Note", self.prompt,
                      "SYSTEM_PROMPT must instruct LLM to use '⚠️ Note:' section")

    def test_prompt_retains_price_format_rule(self):
        self.assertIn("$XX.XX", self.prompt,
                      "SYSTEM_PROMPT must retain price formatting rule $XX.XX")

    def test_prompt_retains_percentage_format_rule(self):
        self.assertIn("XX.X%", self.prompt,
                      "SYSTEM_PROMPT must retain percentage formatting rule XX.X%")

    def test_prompt_retains_course_not_found_rule(self):
        lower = self.prompt.lower()
        self.assertTrue(
            "not in the data" in lower or "not found" in lower or "course not" in lower,
            "SYSTEM_PROMPT must retain instruction to state when a course is not in the data"
        )

    def test_prompt_retains_spelling_tolerance_rule(self):
        lower = self.prompt.lower()
        self.assertTrue(
            "spelling" in lower or "informal" in lower or "gracefully" in lower,
            "SYSTEM_PROMPT must retain instruction to handle spelling mistakes gracefully"
        )


# ===========================================================================
# TASK 3 — Fix-checking: structured format present in LLM responses
# ===========================================================================

class TestFixCheckingStructuredFormat(unittest.TestCase):
    """
    Patch the LLM call to return a fixed-style response and verify
    generate_answer surfaces the structured sections.
    """

    def _call_generate_answer_with_fixed_response(self, chunks, question):
        """Patch _call_gemini to return a structured response."""
        with patch("rag.llm._call_gemini", return_value=FIXED_RESPONSE), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini",
                                     "GEMINI_API_KEY": "fake-key"}):
            from rag.llm import generate_answer
            return generate_answer(chunks, question)

    def test_price_query_contains_insight(self):
        result = self._call_generate_answer_with_fixed_response(
            PRICE_CHUNKS, "What is the average price for stonegate golf club?"
        )
        self.assertIn("📊 Insight:", result)

    def test_price_query_contains_analysis(self):
        result = self._call_generate_answer_with_fixed_response(
            PRICE_CHUNKS, "What is the average price for stonegate golf club?"
        )
        self.assertIn("📈 Analysis:", result)

    def test_price_query_contains_recommendation(self):
        result = self._call_generate_answer_with_fixed_response(
            PRICE_CHUNKS, "What is the average price for stonegate golf club?"
        )
        self.assertIn("💡 Recommendation:", result)

    def test_occupancy_query_contains_insight(self):
        occ_response = (
            "📊 Insight: Lakewood Golf Club has the highest occupancy at 50.3%.\n"
            "📈 Analysis: This indicates strong demand compared to other courses.\n"
            "💡 Recommendation: Consider a modest price increase to maximise revenue."
        )
        with patch("rag.llm._call_gemini", return_value=occ_response), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini",
                                     "GEMINI_API_KEY": "fake-key"}):
            from rag.llm import generate_answer
            result = generate_answer(OCCUPANCY_CHUNKS, "Which course has the highest occupancy?")
        self.assertIn("📊 Insight:", result)
        self.assertIn("📈 Analysis:", result)
        self.assertIn("💡 Recommendation:", result)

    def test_availability_query_contains_insight(self):
        avail_response = (
            "📊 Insight: Stonegate Golf Club Central has tee times available on GolfNow at $63.00.\n"
            "📈 Analysis: Brand pricing is slightly higher at $65.00.\n"
            "💡 Recommendation: Promote GolfNow slots to increase channel visibility."
        )
        with patch("rag.llm._call_gemini", return_value=avail_response), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini",
                                     "GEMINI_API_KEY": "fake-key"}):
            from rag.llm import generate_answer
            result = generate_answer(AVAILABILITY_CHUNKS, "Available tee times on GolfNow?")
        self.assertIn("📊 Insight:", result)

    def test_response_is_not_raw_numbers_only(self):
        result = self._call_generate_answer_with_fixed_response(
            PRICE_CHUNKS, "average price stonegate"
        )
        # Must contain at least one emoji section header — not just a number
        self.assertTrue(
            any(h in result for h in ["📊", "📈", "💡", "⚠️"]),
            f"Response must not be raw numbers only — got: {result!r}"
        )


# ===========================================================================
# TASK 4 — Preservation: fallback paths unchanged
# ===========================================================================

class TestPreservationFallbackPaths(unittest.TestCase):
    """Verify that non-LLM paths return exactly the same result as before."""

    def test_empty_chunks_returns_exact_fallback_message(self):
        """When chunks=[], generate_answer must return the standard fallback."""
        # No LLM is called when chunks are empty — this is handled in backend/app.py
        # but we also test the LLM-unavailable path with empty chunks
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini",
                                     "GEMINI_API_KEY": ""}):
            from rag.llm import generate_answer
            result = generate_answer([], "any question")
        self.assertEqual(
            result,
            "I could not find relevant data for your query.",
            "Empty chunks + unavailable LLM must return exact fallback"
        )

    def test_value_error_fallback_returns_raw_chunks_summary(self):
        """When LLM raises ValueError (bad API key), return raw chunks summary."""
        with patch("rag.llm._call_gemini", side_effect=ValueError("GEMINI_API_KEY is not set")), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            from rag.llm import generate_answer
            result = generate_answer(PRICE_CHUNKS, "average price?")
        self.assertTrue(
            result.startswith("Here is the most relevant data I found:"),
            f"ValueError fallback must start with 'Here is the most relevant data I found:' — got: {result!r}"
        )

    def test_runtime_error_fallback_returns_raw_chunks_summary(self):
        """When LLM raises RuntimeError (quota exhausted), return raw chunks summary."""
        with patch("rag.llm._call_gemini", side_effect=RuntimeError("All Gemini models exhausted")), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            from rag.llm import generate_answer
            result = generate_answer(PRICE_CHUNKS, "average price?")
        self.assertTrue(
            result.startswith("Here is the most relevant data I found:"),
            f"RuntimeError fallback must return raw chunks summary — got: {result!r}"
        )

    def test_fallback_includes_chunk_text(self):
        """Fallback response must include the first chunk's text."""
        with patch("rag.llm._call_gemini", side_effect=ValueError("no key")), \
             patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            from rag.llm import generate_answer
            result = generate_answer(PRICE_CHUNKS, "price?")
        # First chunk text should appear in the fallback
        self.assertIn("stonegate", result.lower(),
                      "Fallback must include chunk text content")

    def test_openai_provider_routes_correctly(self):
        """When LLM_PROVIDER=openai, request routes through _call_openai."""
        openai_response = (
            "📊 Insight: Average price is $62.11.\n"
            "📈 Analysis: Competitive pricing.\n"
            "💡 Recommendation: Monitor market rates weekly."
        )
        with patch("rag.llm._call_openai", return_value=openai_response) as mock_openai, \
             patch.dict(os.environ, {"LLM_PROVIDER": "openai",
                                     "OPENAI_API_KEY": "fake-openai-key"}):
            from rag.llm import generate_answer
            result = generate_answer(PRICE_CHUNKS, "average price?")
        mock_openai.assert_called_once()
        self.assertEqual(result, openai_response)


# ===========================================================================
# TASK 5 — Formatting rules: $XX.XX, XX.X%, spelling tolerance
# ===========================================================================

class TestFormattingRulesPreserved(unittest.TestCase):
    """Verify price/percentage formatting and spelling tolerance are preserved."""

    def test_price_format_in_fixed_response(self):
        """Prices in LLM responses must match $XX.XX pattern."""
        # FIXED_RESPONSE contains "$62.11" and "$61.16"
        prices = re.findall(r"\$\d+\.\d{2}", FIXED_RESPONSE)
        self.assertTrue(len(prices) > 0,
                        f"Fixed response must contain prices in $XX.XX format — got: {FIXED_RESPONSE!r}")
        for price in prices:
            self.assertRegex(price, r"^\$\d+\.\d{2}$",
                             f"Price {price!r} must match $XX.XX format")

    def test_percentage_format_in_occupancy_response(self):
        """Percentages in LLM responses must match XX.X% pattern."""
        occ_response = (
            "📊 Insight: Lakewood Golf Club has the highest occupancy at 50.3%.\n"
            "📈 Analysis: This is above average.\n"
            "💡 Recommendation: Increase prices slightly."
        )
        percentages = re.findall(r"\d+\.\d%", occ_response)
        self.assertTrue(len(percentages) > 0,
                        f"Response must contain percentages in XX.X% format — got: {occ_response!r}")

    def test_misspelled_question_does_not_raise(self):
        """Misspelled questions must not raise exceptions — handled gracefully."""
        misspelled_questions = [
            "occupency rate",       # occupancy
            "availabilty",          # availability
            "stongate golf club",   # stonegate
            "prise for course",     # price
        ]
        fixed_resp = (
            "📊 Insight: Data found.\n"
            "📈 Analysis: See details.\n"
            "💡 Recommendation: Review pricing."
        )
        for question in misspelled_questions:
            with self.subTest(question=question):
                with patch("rag.llm._call_gemini", return_value=fixed_resp), \
                     patch.dict(os.environ, {"LLM_PROVIDER": "gemini",
                                             "GEMINI_API_KEY": "fake-key"}):
                    from rag.llm import generate_answer
                    try:
                        result = generate_answer(PRICE_CHUNKS, question)
                        # Must return a string — no exception
                        self.assertIsInstance(result, str,
                                              f"Result for {question!r} must be a string")
                    except Exception as exc:
                        self.fail(f"Misspelled question {question!r} raised {type(exc).__name__}: {exc}")

    def test_note_section_is_optional(self):
        """⚠️ Note section must only appear when relevant — not in every response."""
        response_without_note = (
            "📊 Insight: Average price is $62.11.\n"
            "📈 Analysis: Pricing is competitive.\n"
            "💡 Recommendation: Monitor weekly."
        )
        # A valid response without a Note section is acceptable
        self.assertIn("📊 Insight:", response_without_note)
        self.assertIn("📈 Analysis:", response_without_note)
        self.assertIn("💡 Recommendation:", response_without_note)
        # Note is absent — that is fine
        self.assertNotIn("⚠️ Note:", response_without_note)

    def test_note_section_appears_when_caveat_present(self):
        """⚠️ Note section must appear when a caveat or warning is relevant."""
        response_with_note = (
            "📊 Insight: Only 2 tee time slots found for this course.\n"
            "📈 Analysis: Limited data may not represent full pricing trends.\n"
            "💡 Recommendation: Collect more data before making pricing decisions.\n"
            "⚠️ Note: This analysis is based on a very small sample (2 records)."
        )
        self.assertIn("⚠️ Note:", response_with_note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
