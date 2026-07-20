# Bugfix Requirements Document

## Introduction

The chatbot's RAG pipeline — covering dataset chunking/indexing (`rag/ingest.py`), FAISS-based retrieval (`rag/retriever.py`), natural language processing (`rag/nlp.py`), and the deterministic data assistant (`rag/data_assistant.py`) — produces wrong or missing results even when the requested data exists in the dataset. The most visible symptom is the chatbot returning "No data available" for valid date, course, occupancy, and price queries. Root causes include: rows and dates being lost or misrepresented during chunking; date formats like "07-07-2025", "7th july", and "july 7" not being parsed correctly; time components in the dataset's date column causing exact-match failures; retrieval returning irrelevant chunks; and a complete absence of debug logging to diagnose failures. This bugfix stabilises the full pipeline so that any valid dataset query returns accurate, complete results without breaking existing working features.

---

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the dataset is ingested via `rag/ingest.py` THEN the system produces chunks that may omit rows or misrepresent date values, causing those rows to be unreachable by any subsequent retrieval or direct lookup.

1.2 WHEN a user submits a date query using a numeric day-first format such as "07-07-2025" or "07/07/2025" THEN the system fails to parse the date correctly (e.g., interprets it as month-first) and returns "No data available" or data for the wrong date.

1.3 WHEN a user submits a date query using a natural language format such as "7th july", "july 7", or "7 july" THEN the system fails to extract the correct date and returns "No data available" or an unrelated result even when matching data exists.

1.4 WHEN the dataset's `tee_date` column contains datetime values with a time component (e.g., "2026-07-07 00:00:00") THEN the system fails to match them against a date-only query string (e.g., "2026-07-07"), causing exact-match lookups to return zero rows and triggering a false "No data available" response.

1.5 WHEN a user submits a query for a specific course name, occupancy level, or price range THEN the FAISS retriever returns chunks that do not contain the requested data, causing the LLM to produce an incorrect or empty answer.

1.6 WHEN the chatbot returns "No data available" THEN the system provides no log output indicating what date was parsed, how many chunks were retrieved, or how many dataset rows matched, making it impossible to diagnose the failure.

1.7 WHEN a user submits a query that matches data in the dataset THEN the system sometimes returns "No data available" because the deterministic data assistant (`answer_data_query`) and the retrieval path (`retrieve` + `generate_answer`) are not coordinated to search the full dataset before giving up.

---

### Expected Behavior (Correct)

2.1 WHEN the dataset is ingested via `rag/ingest.py` THEN the system SHALL produce one chunk per row for both the availability and market rates CSVs, preserving all rows and all date values in a normalised date-only format (YYYY-MM-DD), so that no row is lost or unreachable.

2.2 WHEN a user submits a date query using a numeric day-first format such as "07-07-2025" or "07/07/2025" THEN the system SHALL parse the date using day-first interpretation and return all rows that exactly match the resolved ISO date "2025-07-07".

2.3 WHEN a user submits a date query using a natural language format such as "7th july", "july 7", or "7 july" (without an explicit year) THEN the system SHALL extract the correct month and day, resolve the year from the dataset's available dates, and return all rows that exactly match the resolved date.

2.4 WHEN the dataset's `tee_date` column is loaded in `rag/data_assistant.py` and `rag/ingest.py` THEN the system SHALL strip any time component by converting to date-only values, ensuring that date-only comparisons always succeed regardless of the original time value stored in the CSV.

2.5 WHEN a user submits a query for a specific course name, occupancy level, or price range THEN the retriever SHALL return chunks that contain data directly relevant to the query, and the LLM SHALL produce an answer that accurately reflects the retrieved data.

2.6 WHEN any date query is processed THEN the system SHALL emit debug log entries recording: the raw query, the parsed date string, the number of chunks retrieved, and the number of dataset rows matched, so that failures can be diagnosed without code changes.

2.7 WHEN a user submits a query that matches data in the dataset THEN the system SHALL search the full dataset (both market rates and availability) before concluding that no data is available, and SHALL return "No data available for [YYYY-MM-DD]" only when zero rows match after a complete search.

---

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user submits a query containing a full ISO date string such as "2026-05-08" THEN the system SHALL CONTINUE TO extract that date correctly and return all matching rows.

3.2 WHEN a user submits a query containing a date with an explicit year in natural language such as "7 July 2026" or "July 7 2026" THEN the system SHALL CONTINUE TO parse and match that date exactly as before.

3.3 WHEN a user submits a query that contains no date reference THEN the system SHALL CONTINUE TO return results filtered only by course name, metric type, or other non-date criteria without any date filtering applied.

3.4 WHEN a user submits a small-talk query such as "hi", "hello", "how are you", "I am fine", or "good morning" THEN the system SHALL CONTINUE TO return the appropriate conversational response without attempting date extraction or data retrieval.

3.5 WHEN a user submits a query for highest or lowest occupancy or price without specifying a date THEN the system SHALL CONTINUE TO return the correctly ranked results across all dates in the dataset.

3.6 WHEN a user submits a query for a specific course name without specifying a date THEN the system SHALL CONTINUE TO return all available data for that course across all dates.

3.7 WHEN a valid date is extracted and matching rows exist in the dataset THEN the system SHALL CONTINUE TO display ALL matching rows (from both the market rates and availability datasets) in a clean, scrollable HTML table without applying a row limit.

3.8 WHEN the LLM response formatting path is used (structured `📊 Insight` / `📈 Analysis` / `💡 Recommendation` sections) THEN the system SHALL CONTINUE TO produce responses in that structured format, and the existing `test_llm_formatting.py` tests SHALL CONTINUE TO pass.

3.9 WHEN the `/ingest/reload` endpoint is called THEN the system SHALL CONTINUE TO rebuild the FAISS index from the CSV files and reload it into the running retriever without restarting the server.

3.10 WHEN the chatbot widget (`chatbot/widget.html`) is served via the `/widget` endpoint THEN the system SHALL CONTINUE TO serve the widget HTML unchanged, with no UI or frontend behaviour modified by this fix.
