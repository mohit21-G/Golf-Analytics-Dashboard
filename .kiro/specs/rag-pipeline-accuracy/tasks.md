# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - RAG Pipeline Date Query Failure
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the five interconnected defects
  - **Scoped PBT Approach**: Scope the property to the three concrete failing cases to ensure reproducibility:
    1. Day-first numeric date: `answer_data_query("show data for 07-05-2026")` where `2026-05-07` exists in the dataset — assert result contains `"2026-05-07"` and `"<table"` (will fail if month-first parsing returns wrong date)
    2. Time-component mismatch: load the CSV, confirm a row exists whose raw `tee_date` contains a time component (e.g. `"2026-05-08 00:00:00"`), call `answer_data_query("data for 2026-05-08")`, assert the row is found (will fail if time component is not stripped in `ingest.py`)
    3. FAISS retrieval date enrichment: call `retrieve("prices on 07-05-2026")` and assert at least one returned chunk has `chunk['date'] == "2026-05-07"` (will fail because the query vector does not encode the resolved ISO date)
  - Create `tests/test_rag_pipeline.py` with a `TestBugConditionExploration` class
  - Use `hypothesis` or `pytest-parametrize` to generate (day, month, year) triples from dates present in the dataset; for each, construct `f"{day:02d}-{month:02d}-{year}"` and assert `answer_data_query` returns data containing the correct ISO date string
  - Run test on UNFIXED code — **EXPECTED OUTCOME**: Test FAILS (this is correct — it proves the bug exists)
  - Document counterexamples found (e.g. `"show data for 07-05-2026"` returns `"No data available for 2026-07-05"` instead of data for `2026-05-07`)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Buggy Query Behavior Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (cases where `isBugCondition` returns false):
    - Observe: `answer_data_query("data for 2026-05-08")` returns a result containing `"2026-05-08"` and `"<table"` on unfixed code
    - Observe: `answer_data_query("show 7 July 2026 data")` returns a result containing `"2026-07-07"` on unfixed code
    - Observe: `answer_data_query("highest occupancy")` returns a non-None result containing `"<table"` on unfixed code
    - Observe: `answer_data_query("hi")` returns `"Hello! 😊 How can I help you?"` on unfixed code
    - Observe: `answer_data_query("how are you")` returns a string containing `"I'm doing great!"` on unfixed code
    - Observe: `answer_data_query("I am fine")` returns a string containing `"That's great to hear!"` on unfixed code
    - Observe: `answer_data_query("show data for stonegate")` returns a non-None result containing `"Stonegate"` on unfixed code
  - Write property-based tests in `tests/test_rag_pipeline.py` capturing observed behavior patterns from Preservation Requirements:
    - **PBT Preservation — ISO dates**: generate all ISO date strings from the dataset; for each, assert `answer_data_query(f"data for {iso}")` returns a result containing that ISO date and `"<table"`
    - **PBT Preservation — no-date queries**: for each of `["highest occupancy", "lowest price", "show all courses"]`, assert `answer_data_query` returns a non-None result containing `"<table"`
    - **PBT Preservation — small-talk**: assert exact string matches for `"hi"`, `"how are you"`, `"I am fine"`, `"good morning"` responses
    - **PBT Preservation — course queries**: for each course name in the dataset, assert `answer_data_query(f"show data for {course}")` returns a non-None result containing the course display name
  - Run tests on UNFIXED code — **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [ ] 3. Fix RAG pipeline date handling and debug logging

  - [ ] 3.1 Fix `rag/ingest.py` — normalise `tee_date` column before building chunks
    - After loading each CSV with `pd.read_csv()`, apply `pd.to_datetime(df['tee_date'], errors='coerce').dt.date` to strip time components and store back into `df['tee_date']`
    - Update `_availability_row_to_text` to format the date as `row['tee_date'].strftime('%Y-%m-%d')` with a `None` guard (e.g. `str(row['tee_date']) if row['tee_date'] else 'N/A'`)
    - Update `_market_row_to_text` with the same date formatting guard
    - Update `_build_summary_chunks` to format `tee_date` values as `"YYYY-MM-DD"` strings in chunk text (the `data from ... to ...` range line)
    - Set `chunk['date']` to the `"YYYY-MM-DD"` string (not the raw CSV value) for all availability and market row chunks
    - Add debug logging after building chunks: `logger.debug(f"Total chunks: {len(chunks)}, availability: {len(avail_df)}, market: {len(market_df)}")`
    - _Bug_Condition: `isBugCondition(query)` Case B — `dataset_tee_date_has_time_component AND matched_row_count == 0`_
    - _Expected_Behavior: `chunk['date']` matches `r"^\d{4}-\d{2}-\d{2}$"` for every chunk; no time component in chunk text_
    - _Preservation: `build_index()` produces the same number of chunks as before (one per row + summary chunks); no row loss_
    - _Requirements: 2.1, 2.4_

  - [ ] 3.2 Fix `rag/retriever.py` — enrich query vector with resolved ISO date
    - Add `from typing import Optional` import if not already present
    - Add `resolved_date: Optional[str] = None` parameter to `retrieve(query, top_k, resolved_date)`
    - After NLP query expansion (step 1), if `resolved_date` is not `None`, append it to the expanded query: `expanded_query = expanded_query + " " + resolved_date`
    - If `resolved_date` is `None`, attempt to extract a date from the query by importing `_extract_date` from `rag.data_assistant` and calling it with the query; if a date is resolved, append it to `expanded_query`
    - Add debug logging: `logger.debug(f"Resolved date for retrieval: {resolved_date!r}")`, `logger.debug(f"Expanded query: {expanded_query!r}")`, `logger.debug(f"Retrieved {len(results)} chunks after filtering")`
    - _Bug_Condition: `isBugCondition(query)` Case D — `retrieve(query)` does not contain chunks for `parsed_date`_
    - _Expected_Behavior: `retrieve("prices on 07-05-2026")` returns at least one chunk with `chunk['date'] == "2026-05-07"`_
    - _Preservation: `retrieve` with no date in query behaves identically to before; `add_chunks` is unaffected_
    - _Requirements: 2.5, 2.6_

  - [ ] 3.3 Add debug logging to `rag/data_assistant.py`
    - In `answer_data_query`, after `_extract_date` is called, add `logger.debug` calls:
      - `logger.debug(f"[answer_data_query] raw_query={question!r} parsed_date={date_iso!r}")`
      - After date filtering: `logger.debug(f"[answer_data_query] market_rows_matched={len(market_df[market_df['tee_date_iso'] == date_iso])} av_rows_matched={len(av_df[av_df['tee_date_iso'] == date_iso])}")`
      - After building `combined`: `logger.debug(f"[answer_data_query] combined_rows_after_date_filter={len(combined)}")`
    - No functional changes to `_extract_date` — the existing implementation already handles all required formats correctly (confirmed by existing `test_data_assistant.py` tests)
    - _Bug_Condition: `isBugCondition(query)` Case B — silent failure with no log output_
    - _Expected_Behavior: `LOG_LEVEL=DEBUG` output contains parsed date, chunk count, and row count for any date query_
    - _Preservation: All existing `test_data_assistant.py` assertions continue to pass unchanged_
    - _Requirements: 2.6_

  - [ ] 3.4 Add debug logging to `backend/app.py`
    - In the `chat` endpoint, after `answer_data_query(question)` returns, add:
      - `logger.debug(f"[chat] deterministic_answer={'found' if deterministic_answer is not None else 'none'} for question={question!r}")`
    - No structural changes to the endpoint flow — the existing deterministic-first, then FAISS-retrieval order is correct
    - _Bug_Condition: `isBugCondition(query)` — silent failure with no log output_
    - _Expected_Behavior: Debug log shows whether deterministic path found data_
    - _Preservation: All endpoint behavior, response schemas, and CORS settings unchanged_
    - _Requirements: 2.6, 2.7_

  - [ ] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - RAG Pipeline Date Query Returns Correct Data
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior: day-first numeric dates parse correctly, time-component rows are found, FAISS retrieval returns date-matched chunks
    - Run bug condition exploration test from step 1 (`TestBugConditionExploration` in `tests/test_rag_pipeline.py`)
    - **EXPECTED OUTCOME**: Test PASSES (confirms all five defects are fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-Buggy Query Behavior Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run preservation property tests from step 2 (`TestPreservation` in `tests/test_rag_pipeline.py`)
    - Also run the full existing test suites: `pytest tests/test_data_assistant.py tests/test_llm_formatting.py`
    - **EXPECTED OUTCOME**: All tests PASS (confirms no regressions in ISO date queries, no-date queries, small-talk, LLM formatting, and endpoint behavior)
    - Confirm all assertions in `tests/test_data_assistant.py` still pass without modification

- [ ] 4. Write unit tests for `_extract_date` format variants and chunk integrity
  - Add `TestExtractDateFormats` class in `tests/test_rag_pipeline.py`:
    - ISO format: `_extract_date("data for 2026-05-08")` → `"2026-05-08"`
    - Day-first numeric with hyphen: `_extract_date("07-05-2026")` → `"2026-05-07"`
    - Day-first numeric with slash: `_extract_date("07/05/2026")` → `"2026-05-07"`
    - Natural language with year: `_extract_date("7 July 2026")` → `"2026-07-07"`
    - Natural language month-day with year: `_extract_date("July 7 2026")` → `"2026-07-07"`
    - Natural language without year (ordinal): `_extract_date("7th july", available_dates={"2026-07-07"})` → `"2026-07-07"`
    - Natural language without year (month-day): `_extract_date("july 7", available_dates={"2026-07-07"})` → `"2026-07-07"`
    - Natural language without year (day-month): `_extract_date("7 july", available_dates={"2026-07-07"})` → `"2026-07-07"`
  - Add `TestChunkIntegrity` class in `tests/test_rag_pipeline.py`:
    - Call `build_index(force=True)`, load `chunks.pkl`, assert every chunk with a `'date'` key matches `r"^\d{4}-\d{2}-\d{2}$"` (no time component)
    - Assert total chunk count equals `len(avail_df) + len(market_df) + summary_chunk_count` (no row loss)
    - Assert `retrieve("data for 2026-05-08", resolved_date="2026-05-08")` returns at least one chunk with `chunk['date'] == "2026-05-08"`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 5. Write integration tests for `/chat` and `/ingest/reload` endpoints
  - Add `TestIntegration` class in `tests/test_rag_pipeline.py` using `fastapi.testclient.TestClient`:
    - Day-first numeric date query: POST `/chat` with `{"question": "show data for 07-05-2026"}` — assert response `answer` contains `"2026-05-07"` and `"<table"`
    - Natural language date query: POST `/chat` with `{"question": "show 7 july data"}` — assert response `answer` is not `"No data available"` and contains `"<table"` or a valid date
    - `/ingest/reload` endpoint: POST `/ingest/reload` — assert response `status == "rebuilt"` and `chunks_added > 0`; then POST `/chat` with a valid ISO date query and assert the response contains matching data
    - Widget endpoint: GET `/widget` — assert HTTP 200 and `Content-Type: text/html`
  - _Requirements: 2.1, 2.2, 2.3, 2.7, 3.9, 3.10_

- [ ] 6. Rebuild the FAISS index by running `rag/ingest.py` after fixes
  - After all code fixes are applied, run `python -m rag.ingest --force` to rebuild the FAISS index from the updated `ingest.py`
  - Verify the rebuild completes without errors and logs `"Index saved"` with a non-zero vector count
  - Verify the rebuilt `rag/index/chunks.pkl` contains no chunks with time components in the `'date'` field
  - Verify the rebuilt index is loaded correctly by the retriever on the next request
  - _Requirements: 2.1, 2.4, 3.9_

- [ ] 7. Checkpoint — Ensure all tests pass
  - Run the full test suite: `pytest tests/test_data_assistant.py tests/test_llm_formatting.py tests/test_rag_pipeline.py -v`
  - Confirm all existing tests in `test_data_assistant.py` pass without modification
  - Confirm all existing tests in `test_llm_formatting.py` pass without modification
  - Confirm all new tests in `test_rag_pipeline.py` pass (exploration, preservation, unit, integration)
  - Ensure all tests pass; ask the user if questions arise.
