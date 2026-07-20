# Bugfix Requirements Document

## Introduction

The chatbot's date understanding system fails to reliably detect and match dates expressed in natural language. Users querying with phrases like "7th July", "July 7", "07-07-2026", "I want 7th july data", or "show details for july 7" either receive wrong results or a "No data available" message even when matching data exists in the dataset. Additionally, the system sometimes returns rows for nearby/approximate dates instead of the exact requested date, and the dataset's date column is not always normalised to date-only values before matching, causing time-component mismatches. This bugfix addresses all four failure modes to make the chatbot fully reliable for date-based queries.

---

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user submits a query containing a natural language date such as "7th july", "July 7", or "7 July" (without an explicit year) THEN the system fails to extract the correct date and returns "No data available" or an unrelated result even when data for that date exists in the dataset.

1.2 WHEN a user submits a query containing a numeric date in day-first format such as "07-07-2026" or "07/07/2026" THEN the system incorrectly parses the date (e.g., interprets it as month-first) and returns data for the wrong date or "No data available".

1.3 WHEN a user submits a query with extra surrounding words such as "I want 7th july data" or "show details for july 7" THEN the system fails to extract the embedded date and returns no results or an incorrect fallback response.

1.4 WHEN the dataset's `tee_date` column contains datetime values with a time component (e.g., "2026-07-07 00:00:00") THEN the system fails to match them against a date-only query string (e.g., "2026-07-07"), causing exact-match lookups to return zero rows and triggering a false "No data available" response.

1.5 WHEN no exact date match is found in the dataset THEN the system returns rows for the nearest or approximate date instead of returning "No data available for [date]", producing misleading results.

---

### Expected Behavior (Correct)

2.1 WHEN a user submits a query containing a natural language date such as "7th july", "July 7", or "7 July" (without an explicit year) THEN the system SHALL extract the correct month and day, resolve the year from the dataset's available dates, and return all rows that exactly match the resolved date.

2.2 WHEN a user submits a query containing a numeric date in day-first format such as "07-07-2026" or "07/07/2026" THEN the system SHALL parse the date using day-first interpretation and return all rows that exactly match "2026-07-07".

2.3 WHEN a user submits a query with extra surrounding words such as "I want 7th july data" or "show details for july 7" THEN the system SHALL extract the embedded date from the query, ignoring non-date words, and return all rows that exactly match the extracted date.

2.4 WHEN the dataset's `tee_date` column is loaded THEN the system SHALL convert it using `pd.to_datetime(df["tee_date"]).dt.date` to strip any time component, ensuring that date-only comparisons always succeed regardless of the original time value stored in the CSV.

2.5 WHEN a date is successfully extracted from the user query but no rows in the dataset exactly match that date THEN the system SHALL return "No data available for [YYYY-MM-DD]" without returning any approximate or nearby-date rows.

---

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user submits a query containing a full ISO date string such as "2026-05-08" THEN the system SHALL CONTINUE TO extract that date correctly and return all matching rows.

3.2 WHEN a user submits a query containing a date with an explicit year in natural language such as "7 July 2026" or "July 7 2026" THEN the system SHALL CONTINUE TO parse and match that date exactly as before.

3.3 WHEN a user submits a query that contains no date reference THEN the system SHALL CONTINUE TO return results filtered only by course name, metric type, or other non-date criteria without any date filtering applied.

3.4 WHEN a user submits a small-talk query such as "hi", "hello", or "how are you" THEN the system SHALL CONTINUE TO return the appropriate conversational response without attempting date extraction.

3.5 WHEN a user submits a query for highest or lowest occupancy or price without specifying a date THEN the system SHALL CONTINUE TO return the ranked results across all dates in the dataset.

3.6 WHEN a user submits a query for a specific course name without specifying a date THEN the system SHALL CONTINUE TO return all available data for that course across all dates.

3.7 WHEN a valid date is extracted and matching rows exist in the dataset THEN the system SHALL CONTINUE TO display ALL matching rows (from both the market rates and availability datasets) in a clean, scrollable HTML table with proper horizontal and vertical alignment, without applying a row limit.
