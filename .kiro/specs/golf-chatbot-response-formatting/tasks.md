# Implementation Tasks — Golf Chatbot Response Formatting Bugfix

## Tasks

- [ ] 1. Write bug condition exploration property test
  - Create `tests/test_llm_formatting.py`
  - Write a test that calls `generate_answer` with sample chunks and asserts the response does NOT yet contain `📊 Insight:` — confirms bug exists on unfixed code
  - Run on unfixed code to capture counterexample

- [ ] 2. Replace SYSTEM_PROMPT in rag/llm.py with structured formatting instructions
  - Replace the `SYSTEM_PROMPT` constant with one that instructs the LLM to always use the four-section format: `📊 Insight:` / `📈 Analysis:` / `💡 Recommendation:` / `⚠️ Note:` (Note only when relevant)
  - Retain all existing rules: `$XX.XX` prices, `XX.X%` percentages, spelling tolerance, course-not-found handling
  - Do NOT change any function logic

- [ ] 3. Write fix-checking tests to verify structured format is applied
  - Assert responses contain `📊 Insight:`, `📈 Analysis:`, `💡 Recommendation:`
  - Cover price, occupancy, and availability question types

- [ ] 4. Write preservation tests to verify fallback paths are unchanged
  - Assert empty chunks returns exact fallback message
  - Assert ValueError fallback returns raw chunks summary
  - Assert OpenAI routing works correctly

- [ ] 5. Verify price and percentage formatting rules are preserved
  - Assert prices match `\$\d+\.\d{2}`, percentages match `\d+\.\d%`
  - Assert misspelled questions do not raise exceptions

- [ ] 6. Run all tests and confirm green
