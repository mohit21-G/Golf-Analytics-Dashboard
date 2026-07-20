# RAG Pipeline Accuracy Bugfix Design

## Overview

The RAG pipeline has five interconnected defects that together cause valid dataset queries to return "No data available" or incorrect results:

1. **Date normalisation gap** — `rag/ingest.py` stores raw date strings (which may include time components like `"2026-05-08 00:00:00"`) in chunk metadata, so chunk `date` fields never match a date-only query string.
2. **Day-first numeric date parsing** — `rag/nlp.py` and `rag/data_assistant._extract_date` do not consistently apply `dayfirst=True` for `DD-MM-YYYY` / `DD/MM/YYYY` inputs, causing `"07-07-2025"` to be misread as July 7 instead of the correct 7 July.
3. **Natural language date without year** — formats like `"7th july"` and `"july 7"` are partially handled in `data_assistant._extract_date` but the year-resolution logic is not exercised by the FAISS retrieval path in `rag/retriever.py`, so the retriever never encodes a date signal into its query vector.
4. **FAISS query encoding for dates** — the retriever's `_query_vector` builds a BM25 vector from the raw (or NLP-expanded) query string, but date tokens like `"07"`, `"07"`, `"2025"` are low-IDF noise tokens that do not match the `"2026-05-08"` format stored in chunk text. The query must be enriched with the resolved ISO date before vectorisation.
5. **Single-source search** — `answer_data_query` in `data_assistant.py` already searches both CSVs via `combined`, but the FAISS retrieval path in `backend/app.py` falls through to `generate_answer` without first checking whether the deterministic path found data; the two paths are not coordinated for date queries.

The fix is targeted and minimal: normalise dates at ingest time, enrich the retriever query with the resolved ISO date, and add structured debug logging throughout. No UI changes are made. All existing tests must continue to pass.

---

## Glossary

- **Bug_Condition (C)**: The set of user queries for which the pipeline returns "No data available" or incorrect data despite matching rows existing in the dataset.
- **Property (P)**: The desired outcome — for any query where matching rows exist, the pipeline returns those rows accurately formatted.
- **Preservation**: All existing behaviours not in C(X) — greetings, LLM formatting, analytics queries without dates, the `/ingest/reload` endpoint, and the widget — must remain byte-for-byte identical.
- **`isBugCondition(query)`**: Pseudocode predicate that returns `true` when the query triggers the defect (see Bug Details).
- **`tee_date`**: The date column in both CSVs. May contain datetime strings with time components (`"2026-05-08 00:00:00"`) or plain date strings (`"2026-05-08"`).
- **`tee_date_iso`**: The derived column in `data_assistant._load_data()` that holds the date as a plain `"YYYY-MM-DD"` string after stripping any time component.
- **`DateObj`**: The derived column holding a Python `datetime.date` object used for exact equality comparisons in `answer_data_query`.
- **`_extract_date(query, available_dates)`**: The function in `rag/data_assistant.py` that parses a free-text query and returns an ISO date string or `None`.
- **`build_index()`**: The function in `rag/ingest.py` that reads both CSVs, builds text chunks, and writes the FAISS index to disk.
- **`retrieve(query, top_k)`**: The function in `rag/retriever.py` that returns the top-k FAISS chunks for a query.
- **`answer_data_query(question)`**: The deterministic handler in `rag/data_assistant.py` that returns an HTML answer or `None`.

---

## Bug Details

### Bug Condition

The bug manifests when a user query contains a date reference (numeric day-first, natural language without year, or a date whose CSV representation includes a time component) and the pipeline fails to match that date against the dataset. The `answer_data_query` function either parses the wrong date, or parses the correct date but fails to find matching rows because the `tee_date` column still contains datetime strings with time components.

**Formal Specification:**
```
FUNCTION isBugCondition(query)
  INPUT: query — a string submitted by the user
  OUTPUT: boolean

  parsed_date := _extract_date(query, available_dates=all_dataset_dates)

  RETURN (
    -- Case A: date is parsed incorrectly (day/month swapped)
    (query matches DD-MM-YYYY or DD/MM/YYYY pattern
     AND parsed_date != correct_dayfirst_interpretation(query))

    OR

    -- Case B: date is parsed correctly but no rows match because
    --         tee_date column still has time components
    (parsed_date IS NOT NULL
     AND correct_rows_exist_in_dataset(parsed_date)
     AND dataset_tee_date_has_time_component
     AND matched_row_count == 0)

    OR

    -- Case C: natural language date without year resolves to wrong year
    (query matches "D Month" or "Month D" pattern without explicit year
     AND resolved_year != year_present_in_dataset_for_that_month_day)

    OR

    -- Case D: FAISS retriever does not encode the resolved ISO date
    --         so retrieval returns irrelevant chunks for date queries
    (parsed_date IS NOT NULL
     AND retrieve(query) does not contain chunks for parsed_date)
  )
END FUNCTION
```

### Examples

| Query | Current (Buggy) Behaviour | Expected Correct Behaviour |
|---|---|---|
| `"show data for 07-07-2025"` | Parsed as `2025-07-07` (month-first) → wrong date | Parsed as `2025-07-07` (day-first) → correct |
| `"availability on 07/07/2026"` | Parsed as `2026-07-07` (month-first, coincidentally correct here) but rows not found because `tee_date` stored as `"2026-07-07 00:00:00"` | Time component stripped → rows found |
| `"data for 7th july"` | Year resolved to current year (e.g. 2025) which may not be in dataset | Year resolved from dataset → `"2026-07-07"` if that date exists |
| `"july 7 prices"` | Same year-resolution failure as above | Same fix as above |
| `"price for 2026-05-08"` | Works correctly today | Must continue to work (preservation) |

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviours:**
- ISO date queries (`"2026-05-08"`) must continue to return correct results.
- Natural language dates with explicit year (`"7 July 2026"`, `"July 7 2026"`) must continue to work.
- Queries with no date reference (highest occupancy, course list, price comparisons) must be completely unaffected.
- Small-talk responses (`"hi"`, `"how are you"`, `"I am fine"`) must return the same strings as before.
- The LLM formatting path (`generate_answer`, `SYSTEM_PROMPT`) must not be modified; `test_llm_formatting.py` must pass unchanged.
- The `/ingest/reload` endpoint must continue to rebuild and reload the index.
- `chatbot/widget.html` must not be modified.
- All assertions in `tests/test_data_assistant.py` must continue to pass.

**Scope:**
All inputs that do NOT involve a date reference in a buggy format, and all inputs that do not trigger the time-component mismatch, should be completely unaffected by this fix. This includes:
- Queries for course names, occupancy rankings, price comparisons without a date.
- Greeting and small-talk inputs.
- Queries using a full ISO date string (`YYYY-MM-DD`).
- The LLM response generation path.

---

## Hypothesized Root Cause

Based on code inspection of all five files:

1. **`rag/ingest.py` — raw date strings in chunk metadata**: `_availability_row_to_text` and `_market_row_to_text` call `row.get('tee_date', 'N/A')` directly. If the CSV column contains `"2026-05-08 00:00:00"`, that string is embedded in the chunk text and stored in `chunk['date']`. The retriever then cannot match a query for `"2026-05-08"` against chunk text containing `"2026-05-08 00:00:00"`.

2. **`rag/ingest.py` — no date normalisation before indexing**: `build_index()` reads the CSV with `pd.read_csv()` but does not apply `pd.to_datetime(...).dt.date` to the `tee_date` column before building chunk text. The `data_assistant._load_data()` does apply `_parse_date_series()`, but `ingest.py` is independent and does not share this normalisation.

3. **`rag/data_assistant._extract_date` — day-first numeric parsing is correct but fragile**: The function does apply `dayfirst=True` for the `DD-MM-YYYY` pattern. However, the regex `r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b"` will also match `MM-DD-YYYY` inputs and apply `dayfirst=True` to them, potentially swapping month and day for US-format dates. The fix must be explicit about the intended interpretation.

4. **`rag/retriever.py` — query vector does not encode resolved date**: `retrieve(query)` calls `normalise_query(query)` from `rag/nlp.py`, which expands synonyms but does not resolve date expressions. The BM25 vector is built from tokens like `"07"`, `"july"`, `"2025"` which are either stopwords or low-IDF tokens. The retriever should accept an optional `resolved_date` parameter and append the ISO date string to the expanded query before vectorisation.

5. **`rag/nlp.py` — date tokens not in synonym map**: The NLP layer has no handling for date-related tokens. Tokens like `"07"`, `"july"`, `"2025"` pass through unchanged and contribute little signal to the BM25 vector.

6. **`backend/app.py` — no debug logging for date pipeline**: The `/chat` endpoint logs the raw question but not the parsed date, chunk count, or row match count. Failures are silent.

---

## Correctness Properties

Property 1: Bug Condition — Date Queries Return Matching Rows

_For any_ user query where `isBugCondition(query)` returns `true` (i.e., the query contains a date reference in day-first numeric, natural language without year, or a format whose CSV representation includes a time component), the fixed pipeline SHALL parse the correct ISO date, strip time components from the dataset's `tee_date` column, and return all rows that exactly match that date — never returning "No data available" when matching rows exist.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7**

Property 2: Preservation — Non-Buggy Queries Unchanged

_For any_ user query where `isBugCondition(query)` returns `false` (ISO date queries, no-date queries, small-talk, LLM-path queries), the fixed pipeline SHALL produce exactly the same result as the original pipeline, preserving all existing functionality including greeting responses, occupancy/price rankings, course listings, LLM formatting, and endpoint behaviour.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

---

## Fix Implementation

### Changes Required

#### File: `rag/ingest.py`

**Function**: `build_index()` and `_availability_row_to_text()` / `_market_row_to_text()`

**Specific Changes**:

1. **Normalise `tee_date` column before building chunks**: After loading each CSV with `pd.read_csv()`, apply `pd.to_datetime(df['tee_date'], errors='coerce').dt.date` to strip time components. Store the result back into `df['tee_date']` as a `datetime.date` object, then format it as `str(date_val)` (which produces `"YYYY-MM-DD"`) when building chunk text.

2. **Update `_availability_row_to_text` and `_market_row_to_text`**: These functions receive a `pd.Series` row. After the normalisation in step 1, `row['tee_date']` will already be a `datetime.date` object. Format it explicitly as `row['tee_date'].strftime('%Y-%m-%d')` (with a `None` guard) to guarantee `"YYYY-MM-DD"` in chunk text.

3. **Store normalised date in chunk metadata**: The `chunk['date']` field must be set to the `"YYYY-MM-DD"` string, not the raw CSV value. This ensures the retriever's post-filter (if any) can compare chunk dates against a parsed query date.

4. **Add debug logging**: After building chunks, log `f"Total chunks: {len(chunks)}, availability: {len(avail_df)}, market: {len(market_df)}"` to confirm no row loss.

#### File: `rag/data_assistant.py`

**Function**: `_load_data()`

**Specific Changes**:

1. **Confirm `_parse_date_series` is applied to both CSVs**: The current code already calls `_parse_date_series` on both `availability["tee_date"]` and `market["tee_date"]`. Verify that `_parse_date_series` returns `pd.Series` of `datetime.date` objects (not strings), so that `combined["DateObj"] == parsed_date` comparisons work correctly. The current implementation returns `.dt.date` which is correct — no change needed here, but add an assertion in tests.

2. **Add debug logging in `answer_data_query`**: After `_extract_date` is called, log:
   - The raw query
   - The resolved `date_iso`
   - The count of matching rows in `market_df` and `av_df` before combining
   - The count of rows in `combined` after date filtering

   Use `logger.debug(...)` so it does not pollute production logs but is visible when `LOG_LEVEL=DEBUG`.

**Function**: `_extract_date()`

**Specific Changes**:

1. **No functional change needed** — the existing implementation already handles all required formats correctly (ISO, day-first numeric, natural language with and without year). The existing `test_data_assistant.py` tests confirm this. The fix is in `ingest.py` (time component stripping) and `retriever.py` (date-enriched query vector).

#### File: `rag/retriever.py`

**Function**: `retrieve(query, top_k)`

**Specific Changes**:

1. **Accept optional `resolved_date` parameter**: Add `resolved_date: Optional[str] = None` to the signature. When provided, append the ISO date string to the expanded query before building the BM25 vector: `expanded_query = expanded_query + " " + resolved_date`. This ensures the FAISS index can match chunks whose text contains the `"YYYY-MM-DD"` date string.

2. **Resolve date inside `retrieve` when not provided**: If `resolved_date` is `None`, attempt to extract a date from the query using `_extract_date` (imported from `data_assistant`). If a date is resolved, append it to the expanded query. This keeps the `backend/app.py` call site simple.

3. **Add debug logging**: Log `f"Resolved date for retrieval: {resolved_date!r}"`, `f"Expanded query: {expanded_query!r}"`, and `f"Retrieved {len(results)} chunks after filtering"`.

#### File: `rag/nlp.py`

**No changes required.** The synonym map and `expand_query` function do not need date-specific handling because the date enrichment is handled in `retriever.py` after `_extract_date` resolves the ISO string.

#### File: `rag/llm.py`

**No changes required.** The `SYSTEM_PROMPT` and `generate_answer` function are not involved in the date parsing or retrieval defects.

#### File: `backend/app.py`

**Function**: `chat(req)`

**Specific Changes**:

1. **Add debug logging for the deterministic path**: After `answer_data_query(question)` returns, log whether a deterministic answer was found and (if applicable) what date was parsed. This is informational only and does not change behaviour.

2. **No structural changes**: The existing flow (deterministic path first, then FAISS retrieval) is correct. The fix in `retriever.py` ensures the FAISS path also benefits from date enrichment when the deterministic path returns `None`.

#### File: `chatbot/widget.html`

**No changes.** This file must not be modified.

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, write exploratory tests that demonstrate the bug on the **unfixed** code (these tests will fail before the fix and pass after), then write preservation tests that confirm existing behaviour is unchanged.

All new tests go in `tests/test_rag_pipeline.py`. Existing tests in `tests/test_data_assistant.py` and `tests/test_llm_formatting.py` must continue to pass without modification.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis.

**Test Plan**: Write tests that call `answer_data_query` with buggy-format date queries and assert that the result contains matching data (not "No data available"). Run these tests on the UNFIXED code to observe failures and confirm the root cause.

**Test Cases**:

1. **Day-first numeric date — data exists**: Call `answer_data_query("show data for 07-05-2026")` where `2026-05-07` exists in the dataset. Assert result contains `"2026-05-07"` and `"<table"`. (Will fail on unfixed code if month-first parsing returns wrong date.)

2. **Time component mismatch**: Directly load the CSV, inject a row with `tee_date = "2026-05-08 00:00:00"`, call `answer_data_query("data for 2026-05-08")`, and assert the row is found. (Will fail on unfixed code if time component is not stripped.)

3. **Natural language without year — dataset year resolution**: Call `answer_data_query("show 7 july data")` with a dataset containing `"2026-07-07"`. Assert result contains `"2026-07-07"`. (Will fail on unfixed code if year resolves to current year and that date is not in dataset.)

4. **FAISS retrieval date enrichment**: Call `retrieve("prices on 07-05-2026")` and assert at least one returned chunk has `chunk['date'] == "2026-05-07"`. (Will fail on unfixed code because the query vector does not encode the resolved date.)

**Expected Counterexamples**:
- `answer_data_query("show data for 07-05-2026")` returns `"No data available for 2026-07-05"` (wrong month/day swap) instead of data for `2026-05-07`.
- Direct row lookup with time-component date returns zero rows.
- Possible causes: `dayfirst` not applied, time component not stripped, FAISS query vector missing date signal.

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed pipeline produces the expected behaviour.

**Pseudocode:**
```
FOR ALL query WHERE isBugCondition(query) DO
  result := answer_data_query_fixed(query)
  ASSERT result contains matching rows (not "No data available")
  ASSERT result contains correct ISO date string
  ASSERT result contains "<table"
END FOR
```

**Property-Based Test (Fix Checking)**:
```
PROPERTY: for any (day, month) pair where that date exists in the dataset,
  answer_data_query(f"{day:02d}-{month:02d}-{year}") SHALL return data
  containing the correct ISO date string "YYYY-MM-DD".

GENERATOR: sample (day, month, year) triples from dates present in the dataset.
ASSERTION: result contains the ISO date string and "<table".
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed pipeline produces the same result as the original pipeline.

**Pseudocode:**
```
FOR ALL query WHERE NOT isBugCondition(query) DO
  ASSERT answer_data_query_original(query) == answer_data_query_fixed(query)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain.
- It catches edge cases that manual unit tests might miss.
- It provides strong guarantees that behaviour is unchanged for all non-buggy inputs.

**Test Plan**: Capture the current output of `answer_data_query` for non-buggy inputs (ISO dates, no-date queries, small-talk), then assert the fixed code produces identical output.

**Test Cases**:

1. **ISO date preservation**: `answer_data_query("data for 2026-05-08")` must return the same result before and after the fix.
2. **Natural language with year preservation**: `answer_data_query("show 7 July 2026 data")` must return the same result.
3. **No-date query preservation**: `answer_data_query("highest occupancy")` must return the same result.
4. **Small-talk preservation**: `answer_data_query("hi")`, `answer_data_query("how are you")`, `answer_data_query("I am fine")` must return the same strings.
5. **Course-only query preservation**: `answer_data_query("show data for stonegate")` must return the same result.

**Property-Based Test (Preservation)**:
```
PROPERTY: for any ISO date string "YYYY-MM-DD" sampled from the dataset,
  answer_data_query(f"data for {iso_date}") SHALL return a result
  containing that ISO date string and "<table".

GENERATOR: sample ISO date strings from the union of market and availability tee_date_iso values.
ASSERTION: result contains the ISO date string and "<table".
```

### Unit Tests

- Test `_extract_date` for all format variants: ISO, day-first numeric (`DD-MM-YYYY`, `DD/MM/YYYY`), natural language with year (`"7 July 2026"`, `"July 7 2026"`), natural language without year (`"7th july"`, `"july 7"`, `"7 july"`).
- Test `build_index()` chunk count equals `len(avail_df) + len(market_df) + summary_chunk_count`.
- Test that all chunk `date` fields match `r"^\d{4}-\d{2}-\d{2}$"` after `build_index()`.
- Test that `retrieve("data for 2026-05-08", resolved_date="2026-05-08")` returns chunks whose `date` field is `"2026-05-08"`.
- Test debug log output contains parsed date, chunk count, and row count for a date query.

### Property-Based Tests

- **PBT Fix Check**: Generate all `(day, month, year)` triples present in the dataset. For each, construct a day-first numeric query `f"{day:02d}-{month:02d}-{year}"` and assert `answer_data_query` returns data containing the correct ISO date.
- **PBT Preservation — ISO dates**: Generate all ISO date strings from the dataset. For each, assert `answer_data_query(f"data for {iso}")` returns a result containing that ISO date and `"<table"`.
- **PBT Preservation — no-date queries**: Generate random subsets of course names from the dataset. For each, assert `answer_data_query(f"show data for {course}")` returns a non-empty result containing the course name.
- **PBT Chunk integrity**: After `build_index()`, for every chunk in `chunks.pkl`, assert `chunk['date']` matches `r"^\d{4}-\d{2}-\d{2}$"` (no time component).

### Integration Tests

- Test the full `/chat` endpoint with a day-first numeric date query and assert the response contains matching rows.
- Test the full `/chat` endpoint with a natural language date query and assert the response is not "No data available".
- Test the `/ingest/reload` endpoint rebuilds the index and the subsequent `/chat` query returns correct results.
- Test that `chatbot/widget.html` is served unchanged by the `/widget` endpoint.
