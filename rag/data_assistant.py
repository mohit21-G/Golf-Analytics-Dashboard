"""
Deterministic, data-driven query assistant for golf chatbot.

This module answers common analytics questions directly from CSV data using
dynamic filtering, fuzzy matching, and structured markdown formatting.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
AVAILABILITY_CSV = BASE_DIR / "Availability.csv"
MARKET_RATES_CSV = BASE_DIR / "Market Rates.csv"
_DRIVE_CSV_CACHE = BASE_DIR / "rag" / "index" / "drive_dataset.csv"
logger = logging.getLogger(__name__)


_SMALL_TALK_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood\s*morning\b",
    r"\bgood\s*afternoon\b",
    r"\bgood\s*evening\b",
    r"\bhow\s+are\s+you\b",
    r"\bwhat'?s\s+up\b",
]

_HOW_ARE_YOU_PATTERNS = [
    r"\bhow\s+are\s+you\b",
    r"\bhow\s+are\s+u\b",
    r"\bhow\'s\s+it\s+going\b",
]

# Positive mood replies (include casual variants)
_USER_POSITIVE_PATTERNS = [
    r"\bi\s+am\s+fine\b",
    r"\bi\'m\s+fine\b",
    r"\bi\s+am\s+good\b",
    r"\bi\'m\s+good\b",
    r"\bi\s+am\s+great\b",
    r"\bi\'m\s+great\b",
    r"\bdoing\s+good\b",
    r"\bdoing\s+well\b",
    r"\bokay\b",
    r"\bnot\s+bad\b",
    r"\bpretty\s+good\b",
    r"\bi\s+am\s+ok\b",
]

# Negative mood replies
_USER_NEGATIVE_PATTERNS = [
    r"\bi\s+am\s+not\s+well\b",
    r"\bi\'m\s+not\s+well\b",
    r"\bi\s+am\s+not\s+good\b",
    r"\bi\'m\s+not\s+good\b",
    r"\bnot\s+great\b",
    r"\bnot\s+good\b",
    r"\bsad\b",
    r"\bterrible\b",
    r"\bawful\b",
    r"\bfeeling\s+bad\b",
    r"\bcould\s+be\s+better\b",
]

_SUPERLATIVE_MAX = {
    "top",
    "best",
    "highest",
    "maximum",
    "max",
    "most",
    "peak",
}
_SUPERLATIVE_MIN = {
    "lowest",
    "minimum",
    "min",
    "least",
    "cheapest",
    "bottom",
}

_TERM_CANONICAL = {
    "occupency": "occupancy",
    "ocupancy": "occupancy",
    "occupansy": "occupancy",
    "availabilty": "availability",
    "availibility": "availability",
    "availble": "available",
    "prise": "price",
    "prce": "price",
    "cost": "price",
    "rate": "price",
    "rates": "price",
    "fee": "price",
    "fees": "price",
    "club": "course",
    "clubs": "course",
    "courses": "course",
    "names": "name",
    "stats": "data",
    "details": "data",
}

_KNOWN_TERMS = {
    "occupancy",
    "availability",
    "available",
    "price",
    "course",
    "name",
    "date",
    "highest",
    "lowest",
    "list",
    "show",
    "data",
    "market",
}


@dataclass
class DataStore:
    availability: pd.DataFrame
    market: pd.DataFrame


def _pretty_course_name(raw: str) -> str:
    return str(raw).replace("_", " ").strip().title()


def _to_iso_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse potentially mixed date formats robustly and return date-only values."""
    parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
    parsed_monthfirst = pd.to_datetime(series, errors="coerce", dayfirst=False)
    parsed = parsed_dayfirst.fillna(parsed_monthfirst)
    return parsed.dt.date


def _format_currency(value: float) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    # Show no decimals for whole dollars, else show two decimals
    if abs(v - round(v)) < 1e-9:
        return f"${int(round(v))}"
    return f"${v:.2f}"


def _format_pct(value: float) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    # Prefer integer percent for readability
    return f"{int(round(v))}%"


def _html_table(
    df: pd.DataFrame,
    columns: list[str],
    renames: dict[str, str],
    limit: Optional[int] = None,
    deduplicate: bool = True,
) -> str:
    """
    Render a clean, responsive HTML table for chatbot display.
    All rows are shown by default (limit=None). Pass an explicit limit only
    for summary/insight views where a cap is intentional.
    """
    if df.empty:
        return "<div>No matching rows found in the dataset.</div>"

    # Keep columns present only
    cols = [c for c in columns if c in df.columns]
    view = df.loc[:, cols].copy()

    # Remove duplicate rows only when requested.
    if deduplicate:
        view = view.drop_duplicates(subset=cols)

    total = len(view)
    if limit is not None:
        view = view.head(limit)

    # rename for nicer headers
    view.rename(columns=renames, inplace=True)

    headers = list(view.columns)
    html = [
        "<div style='max-width:100%;overflow-x:auto;overflow-y:auto;max-height:520px;border:1px solid #475569;border-radius:8px;background:#0f172a;'>",
        "<table style='width:100%;border-collapse:collapse;table-layout:auto;font-size:12px;line-height:1.45;'>",
    ]
    # sticky header row
    html.append(
        "<thead><tr>"
        + "".join(
            f"<th style='position:sticky;top:0;z-index:1;background:#1f2937;color:#d1fae5;padding:8px 10px;text-align:left;border:1px solid #475569;white-space:nowrap;'>{h}</th>"
            for h in headers
        )
        + "</tr></thead><tbody>"
    )
    for _, row in view.iterrows():
        html.append(
            "<tr>"
            + "".join(
                f"<td style='padding:7px 10px;border:1px solid #334155;color:#e5e7eb;white-space:nowrap;vertical-align:top;'>{row[h]}</td>"
                for h in headers
            )
            + "</tr>"
        )
    html.append("</tbody></table></div>")

    if limit is not None and total > limit:
        html.append(f"<div style='margin-top:6px;font-size:12px;color:#cbd5e1;'>Showing {min(limit,total)} of {total} matching rows.</div>")

    return ''.join(html)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise_text(text: str) -> str:
    tokens = _tokenize(text)
    normalised: list[str] = []
    for tok in tokens:
        mapped = _TERM_CANONICAL.get(tok, tok)
        if mapped == tok and tok not in _KNOWN_TERMS:
            close = get_close_matches(tok, list(_KNOWN_TERMS), n=1, cutoff=0.84)
            if close:
                mapped = close[0]
        normalised.append(mapped)
    return " ".join(normalised)


@lru_cache(maxsize=1)
def _load_data() -> DataStore:
    availability = pd.read_csv(AVAILABILITY_CSV)
    availability.columns = availability.columns.str.strip().str.lower()

    market = pd.read_csv(MARKET_RATES_CSV)
    market.columns = market.columns.str.strip().str.lower()

    # ── Merge Google Drive dataset (same source as project.py) ────────────────
    if _DRIVE_CSV_CACHE.exists():
        try:
            drive_df = pd.read_csv(_DRIVE_CSV_CACHE)
            drive_df.columns = drive_df.columns.str.strip().str.lower()
            # Only merge rows that share the same schema as market (must have avg_price)
            if "avg_price" in drive_df.columns and "tee_date" in drive_df.columns:
                # Align columns — add any missing columns as NaN so concat works cleanly
                for col in market.columns:
                    if col not in drive_df.columns:
                        drive_df[col] = pd.NA
                drive_df = drive_df[market.columns]
                market = pd.concat([market, drive_df], ignore_index=True).drop_duplicates()
                logger.info(
                    f"Drive dataset merged into market: {len(drive_df)} rows added, "
                    f"total market rows now {len(market)}"
                )
            else:
                logger.warning(
                    "Drive dataset missing 'avg_price' or 'tee_date' column — skipping merge"
                )
        except Exception as exc:
            logger.warning(f"Could not load Drive dataset cache ({exc}) — using local CSVs only")

    # Normalize dataset dates to date-only for exact matching regardless of time parts.
    availability["tee_date"] = _parse_date_series(availability["tee_date"])
    market["tee_date"] = _parse_date_series(market["tee_date"])

    availability["course_name"] = availability["course_name"].astype(str)
    market["course_name"] = market["course_name"].astype(str)

    availability["course_display"] = availability["course_name"].map(_pretty_course_name)
    market["course_display"] = market["course_name"].map(_pretty_course_name)

    availability["tee_date_iso"] = _to_iso_date(availability["tee_date"])
    market["tee_date_iso"] = _to_iso_date(market["tee_date"])

    return DataStore(availability=availability, market=market)


def _is_small_talk(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _SMALL_TALK_PATTERNS)


def _time_based_greeting(now: Optional[datetime] = None) -> str:
    current = now or datetime.now()
    hour = current.hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def _small_talk_response(question: str) -> Optional[str]:
    q = question.lower().strip()
    guidance = "How can I help you today? You can ask about golf data like prices, occupancy, availability, or course details."

    if any(re.search(p, q) for p in _HOW_ARE_YOU_PATTERNS):
        return f"I'm doing great! 😊 How about you?\n\n{guidance}"

    if any(re.search(p, q) for p in _USER_POSITIVE_PATTERNS):
        return f"That's great to hear! 😊 {guidance}"

    if any(re.search(p, q) for p in _USER_NEGATIVE_PATTERNS):
        return f"I'm sorry to hear that. Hope I can help you 😊 What do you need?\n\n{guidance}"

    # Time-based greeting should only be used when user explicitly says
    # good morning / good evening.
    if re.search(r"\bgood\s*morning\b", q) or re.search(r"\bgood\s*evening\b", q):
        return f"{_time_based_greeting()}! 😊 How can I help you?"

    if re.search(r"\b(hi|hello|hey|greetings|what'?s\s*up|sup)\b", q):
        return "Hello! 😊 How can I help you?"

    return None


def _extract_date(question: str, available_dates: Optional[set[str]] = None) -> Optional[str]:
    def _parse_available_date_values() -> list[date]:
        if not available_dates:
            return []
        parsed_values: list[date] = []
        for value in available_dates:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.notna(dt):
                parsed_values.append(dt.date())
        return sorted(set(parsed_values))

    def _dataset_year() -> Optional[int]:
        values = _parse_available_date_values()
        if not values:
            return None
        counts: dict[int, int] = {}
        for value in values:
            counts[value.year] = counts.get(value.year, 0) + 1
        return max(sorted(counts), key=lambda year: counts[year])

    def _safe_date(year: int, month: int, day: int) -> Optional[date]:
        dt = pd.to_datetime(f"{year:04d}-{month:02d}-{day:02d}", errors="coerce")
        if pd.notna(dt):
            return dt.date()
        return None

    def _resolve_yearless(month: int, day: int) -> Optional[str]:
        values = _parse_available_date_values()
        if values:
            exact_month_day = [d for d in values if d.month == month and d.day == day]
            if exact_month_day:
                # Pick the earliest matching dataset date to keep deterministic behavior.
                return min(exact_month_day).strftime("%Y-%m-%d")

            inferred_year = _dataset_year() or datetime.now().year
            inferred_target = _safe_date(inferred_year, month, day)
            if inferred_target:
                return inferred_target.strftime("%Y-%m-%d")

        fallback_year = datetime.now().year
        fallback_target = _safe_date(fallback_year, month, day)
        if fallback_target:
            return fallback_target.strftime("%Y-%m-%d")
        return None

    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", question, flags=re.IGNORECASE)

    iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
    if iso:
        dt = pd.to_datetime(iso.group(0), errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")

    # Numeric date like 07-07-2026 or 07/07/2026
    day_first = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", cleaned)
    if day_first:
        dt = pd.to_datetime(day_first.group(0), errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")

    natural = re.search(r"\b\d{1,2}\s+[a-zA-Z]{3,9}\s+\d{4}\b", cleaned)
    if natural:
        dt = pd.to_datetime(natural.group(0), errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")

    natural_month_day_year = re.search(r"\b[a-zA-Z]{3,9}\s+\d{1,2}\s+\d{4}\b", cleaned)
    if natural_month_day_year:
        dt = pd.to_datetime(natural_month_day_year.group(0), errors="coerce", dayfirst=False)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")

    # Day Month without explicit year (e.g., 7 July)
    day_month = re.search(r"\b(\d{1,2})\s+([a-zA-Z]{3,9})\b", cleaned)
    if day_month:
        day = int(day_month.group(1))
        month_text = day_month.group(2)
        dt = pd.to_datetime(f"{day} {month_text} 2000", errors="coerce", dayfirst=True)
        if pd.notna(dt):
            resolved = _resolve_yearless(int(dt.month), int(dt.day))
            if resolved:
                return resolved

    # Month Day without explicit year (e.g., July 7)
    month_day = re.search(r"\b([a-zA-Z]{3,9})\s+(\d{1,2})\b", cleaned)
    if month_day:
        month_text = month_day.group(1)
        day = int(month_day.group(2))
        dt = pd.to_datetime(f"{day} {month_text} 2000", errors="coerce", dayfirst=True)
        if pd.notna(dt):
            resolved = _resolve_yearless(int(dt.month), int(dt.day))
            if resolved:
                return resolved

    return None


def _match_course(question: str, courses: list[str]) -> Optional[str]:
    q = question.lower().replace("-", " ")
    q_clean = re.sub(r"[^a-z0-9\s]", " ", q)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    for c in courses:
        cname = c.replace("_", " ").lower()
        if cname in q_clean or c.lower() in q_clean:
            return c

    scored: list[tuple[float, str]] = []
    for c in courses:
        cname = c.replace("_", " ").lower()
        ratio = SequenceMatcher(None, q_clean, cname).ratio()
        scored.append((ratio, c))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.52:
        return scored[0][1]
    return None


def _contains_any(text: str, words: set[str]) -> bool:
    toks = set(_tokenize(text))
    return bool(toks & words)


def _top_bottom_insight(df: pd.DataFrame, metric: str, highest: bool) -> str:
    if df.empty:
        return "I could not find matching rows for that request."

    ordered = df.sort_values(metric, ascending=not highest)
    best = ordered.iloc[0]

    metric_label = "Occupancy" if metric == "occ_percent" else "Average Price"
    metric_value = _format_pct(best[metric]) if metric == "occ_percent" else _format_currency(best[metric])
    direction = "highest" if highest else "lowest"

    insight = (
        f"<h3>Insight</h3>"
        f"<div>{_pretty_course_name(best['course_name'])} has the {direction} {metric_label.lower()} at {metric_value}.</div>"
        f"<h3 style='margin-top:8px;'>Top Results</h3>"
    )

    top = ordered.head(10).copy()
    top["avg_price"] = top["avg_price"].map(_format_currency)
    top["occ_percent"] = top["occ_percent"].map(_format_pct)

    insight += _table_from_frame(
        top,
        ["course_display", "tee_date_iso", "avg_price", "occ_percent", "market_avg"],
        limit=10,
    )
    return insight


def _list_courses(market_df: pd.DataFrame) -> str:
    summary = (
        market_df.groupby("course_name", as_index=False)
        .agg(avg_price=("avg_price", "mean"), avg_occupancy=("occ_percent", "mean"))
        .sort_values("course_name")
    )
    summary["course_display"] = summary["course_name"].map(_pretty_course_name)
    summary["avg_price"] = summary["avg_price"].map(_format_currency)
    summary["avg_occupancy"] = summary["avg_occupancy"].map(_format_pct)

    return (
        "<h3>Club Names</h3>"
        + _table_from_frame(
            summary,
            ["course_display", "avg_price", "avg_occupancy"],
        )
    )


def _date_summary(date_iso: str, market_df: pd.DataFrame) -> str:
    day = market_df[market_df["tee_date_iso"] == date_iso].copy()
    if day.empty:
        return f"No rows found for {date_iso}. Try a date within the dataset range."

    day.sort_values("avg_price", ascending=False, inplace=True)
    day["avg_price"] = day["avg_price"].map(_format_currency)
    day["occ_percent"] = day["occ_percent"].map(_format_pct)
    day["market_avg"] = day["market_avg"].map(_format_currency)

    return (
        f"<h3>Data for {date_iso}</h3>"
        + _table_from_frame(
            day,
            ["course_display", "avg_price", "occ_percent", "market_avg"],
        )
    )


def _availability_table(av_df: pd.DataFrame) -> str:
    if av_df.empty:
        return "No matching availability records found."

    cols = [
        "course_display",
        "tee_date_iso",
        "tee_time",
        "brand_availability_status",
        "golfnow_availability_status",
        "overall_availability_status",
    ]
    return (
        "<h3>Availability Snapshot</h3>"
        + _table_from_frame(
            av_df.sort_values(["tee_date", "tee_time"]).copy(),
            cols,
        )
    )


def _available_price(row: pd.Series) -> Optional[float]:
    prices = []
    for key in (
        "brand_current_price",
        "golfnow_current_price",
        "teeoff_current_price",
        "supremegolf_current_price",
    ):
        value = row.get(key)
        if pd.notna(value):
            try:
                prices.append(float(value))
            except (TypeError, ValueError):
                continue
    if not prices:
        return None
    return sum(prices) / len(prices)


def _normalize_market_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    frame = market_df.copy()
    frame["Course Name"] = frame["course_display"]
    frame["Date"] = frame["tee_date_iso"]
    frame["DateObj"] = frame["tee_date"]
    frame["Time"] = ""
    frame["Price"] = frame["avg_price"]
    frame["Occupancy"] = frame["occ_percent"]
    frame["Availability"] = frame.get("overall_availability_status", "")
    frame["Market Avg"] = frame["market_avg"]
    frame["Market Min"] = frame["market_min"]
    frame["Market Max"] = frame["market_max"]
    frame["Brand Price"] = ""
    frame["GolfNow Price"] = ""
    frame["TeeOff Price"] = ""
    frame["SupremeGolf Price"] = ""
    frame["Brand Status"] = ""
    frame["GolfNow Status"] = ""
    frame["TeeOff Status"] = ""
    frame["SupremeGolf Status"] = ""
    frame["Source"] = "market_rates"
    return frame


def _normalize_availability_frame(av_df: pd.DataFrame) -> pd.DataFrame:
    frame = av_df.copy()
    frame["Course Name"] = frame["course_display"]
    frame["Date"] = frame["tee_date_iso"]
    frame["DateObj"] = frame["tee_date"]
    frame["Time"] = frame["tee_time"]
    frame["Price"] = frame.apply(_available_price, axis=1)
    frame["Occupancy"] = ""
    frame["Availability"] = frame["overall_availability_status"]
    frame["Market Avg"] = ""
    frame["Market Min"] = ""
    frame["Market Max"] = ""
    frame["Brand Price"] = frame["brand_current_price"]
    frame["GolfNow Price"] = frame["golfnow_current_price"]
    frame["TeeOff Price"] = frame["teeoff_current_price"]
    frame["SupremeGolf Price"] = frame["supremegolf_current_price"]
    frame["Brand Status"] = frame["brand_availability_status"]
    frame["GolfNow Status"] = frame["golfnow_availability_status"]
    frame["TeeOff Status"] = frame["teeoff_availability_status"]
    frame["SupremeGolf Status"] = frame["supremegolf_availability_status"]
    frame["Source"] = "availability"
    return frame


def _fuzzy_course_filter(frame: pd.DataFrame, question: str, course_name: Optional[str]) -> pd.DataFrame:
    if course_name:
        filtered = frame[frame["course_name"] == course_name]
        if not filtered.empty:
            return filtered

    q = question.lower()
    if not q.strip():
        return frame

    scored: list[tuple[float, int]] = []
    for idx, row in frame.iterrows():
        course_text = str(row.get("course_display", "")).lower()
        row_text = " ".join(str(value) for value in row.values if pd.notna(value)).lower()
        ratio = SequenceMatcher(None, q, course_text).ratio()
        token_hits = sum(1 for token in _tokenize(q) if token and token in row_text)
        score = ratio + (0.15 * token_hits)
        if course_name and row.get("course_name") == course_name:
            score += 2.0
        scored.append((score, idx))

    ranked = [idx for score, idx in sorted(scored, reverse=True) if score > 0.18]
    if not ranked:
        return frame
    return frame.loc[ranked]


def _row_search_score(row: pd.Series, tokens: list[str], date_iso: Optional[str], course_name: Optional[str]) -> float:
    row_text = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
    score = 0.0

    for token in tokens:
        if token and token in row_text:
            score += 1.0

    if course_name and row.get("course_name") == course_name:
        score += 3.0

    if date_iso and str(row.get("Date", "")) == date_iso:
        score += 3.0

    if row.get("Price") is not None and pd.notna(row.get("Price")):
        score += 0.2

    if row.get("Occupancy") is not None and pd.notna(row.get("Occupancy")):
        score += 0.2

    return score


def _infer_columns(question: str, frame: pd.DataFrame) -> list[str]:
    q = question.lower()
    wants_time = any(word in q for word in ["time", "hour", "slot", "tee"])
    wants_market = any(word in q for word in ["market", "compare", "benchmark"])
    wants_availability = any(word in q for word in ["availability", "available", "open", "slot"])
    wants_occupancy = "occupancy" in q
    wants_price = any(word in q for word in ["price", "cost", "rate"])
    wants_course_only = any(word in q for word in ["course", "club", "name"]) and not any(
        word in q for word in ["price", "occupancy", "availability", "available", "market", "time", "hour", "slot", "tee"]
    )

    columns: list[str] = []

    def add(column: str) -> None:
        if column in frame.columns and column not in columns:
            columns.append(column)

    add("Course Name")

    if wants_course_only:
        return columns[:1]

    if wants_time:
        add("Date")
        add("Time")
    else:
        add("Date")

    if wants_price or wants_market:
        add("Price")

    if wants_occupancy:
        add("Occupancy")

    if wants_availability:
        add("Availability")

    if wants_market:
        add("Market Avg")
        add("Market Min")
        add("Market Max")

    if any(word in q for word in ["brand"]):
        add("Brand Price")
        add("Brand Status")
    if any(word in q for word in ["golfnow", "golf now"]):
        add("GolfNow Price")
        add("GolfNow Status")
    if any(word in q for word in ["teeoff", "tee off"]):
        add("TeeOff Price")
        add("TeeOff Status")
    if any(word in q for word in ["supremegolf", "supreme golf"]):
        add("SupremeGolf Price")
        add("SupremeGolf Status")

    if not columns:
        for column in ["Course Name", "Date", "Price", "Occupancy", "Availability"]:
            add(column)

    # No hard cap — return all matched columns
    return columns


def _format_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    view = frame.loc[:, columns].copy()
    for column in view.columns:
        if column in {"Price", "Market Avg", "Market Min", "Market Max", "Brand Price", "GolfNow Price", "TeeOff Price", "SupremeGolf Price"}:
            view[column] = view[column].map(lambda value: _format_currency(value) if pd.notna(value) else "")
        elif column == "Occupancy":
            view[column] = view[column].map(lambda value: _format_pct(value) if pd.notna(value) else "")
        elif column == "Date":
            view[column] = view[column].fillna("").astype(str)
        elif column == "Time":
            view[column] = view[column].fillna("").astype(str)
        else:
            view[column] = view[column].fillna("").astype(str)
    return view


def _table_from_frame(
    frame: pd.DataFrame,
    columns: list[str],
    limit: Optional[int] = None,
    deduplicate: bool = True,
) -> str:
    """Format a dataframe slice as clean HTML table while formatting numbers."""
    if frame.empty:
        return "<div>No matching rows found in the dataset.</div>"

    # Format numeric columns and copy only requested columns
    view = _format_numeric_columns(frame, columns).copy()

    return _html_table(
        view,
        list(view.columns),
        {c: c for c in view.columns},
        limit=limit,
        deduplicate=deduplicate,
    )


def answer_data_query(question: str) -> Optional[str]:
    """
    Return a clean markdown answer when a deterministic data answer is possible.

    Returns None when the query should fall back to retrieval + LLM.
    """
    if not question.strip():
        return "Please ask a question about courses, prices, occupancy, or availability."

    response = _small_talk_response(question)
    if response:
        return response

    if _is_small_talk(question):
        return (
            f"{_time_based_greeting()}! 😊 I'm your Golf Analytics Assistant.\n\n"
            "How can I help you today? You can ask about golf data like prices, occupancy, availability, or course details."
        )

    store = _load_data()
    market_df = store.market.copy()
    av_df = store.availability.copy()

    qn = _normalise_text(question)
    # Use only local CSV dates for year resolution — Drive data may span different
    # years and would skew the "most common year" heuristic in _resolve_yearless.
    local_market = pd.read_csv(MARKET_RATES_CSV)
    local_market.columns = local_market.columns.str.strip().str.lower()
    local_market["tee_date"] = pd.to_datetime(local_market["tee_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local_av_dates = set(av_df["tee_date_iso"].dropna().astype(str))
    local_market_dates = set(local_market["tee_date"].dropna().astype(str))
    available_dates = local_market_dates | local_av_dates
    date_iso = _extract_date(question, available_dates=available_dates)
    parsed_date = pd.to_datetime(date_iso, errors="coerce").date() if date_iso else None
    if date_iso:
        print(f"[date-debug] parsed_date={date_iso}")
        logger.info("Parsed date from user query: %s", date_iso)
        has_exact_market = bool((market_df["tee_date_iso"] == date_iso).any())
        has_exact_availability = bool((av_df["tee_date_iso"] == date_iso).any())
        print(
            f"[date-debug] exact_date_in_dataset market={has_exact_market} "
            f"availability={has_exact_availability}"
        )
        logger.info(
            "Exact dataset date check for %s -> market=%s, availability=%s",
            date_iso,
            has_exact_market,
            has_exact_availability,
        )

    courses = sorted(set(market_df["course_name"].dropna().astype(str)) | set(av_df["course_name"].dropna().astype(str)))
    matched_course = _match_course(qn, courses)

    market_norm = _normalize_market_frame(market_df)
    av_norm = _normalize_availability_frame(av_df)
    combined = pd.concat([market_norm, av_norm], ignore_index=True, sort=False)

    if matched_course:
        combined = combined[combined["course_name"] == matched_course]

    if parsed_date:
        combined = combined[combined["DateObj"] == parsed_date]

    asks_list = any(word in qn for word in ["list", "all", "show"]) and any(
        word in qn for word in ["course", "club", "name"]
    )
    asks_data = any(word in qn for word in ["data", "table", "details", "show", "find", "search"])

    if asks_list:
        courses_df = pd.DataFrame({
            "Course Name": sorted(set(store.market["course_display"]).union(set(store.availability["course_display"]))),
        })
        return _table_from_frame(courses_df, ["Course Name"])

    wants_highest = _contains_any(qn, _SUPERLATIVE_MAX)
    wants_lowest = _contains_any(qn, _SUPERLATIVE_MIN)
    wants_occupancy = "occupancy" in qn
    wants_price = any(word in qn for word in ["price", "cost", "rate", "market"])
    wants_availability = any(word in qn for word in ["availability", "available", "open", "slot"])
    wants_time = any(word in qn for word in ["time", "hour", "tee"])
    wants_market = any(word in qn for word in ["market", "compare", "benchmark"])

    if combined.empty:
        if parsed_date:
            return f"No data available for {date_iso}"
        return "No matching rows found in the dataset. Try a different course name, date, or keyword."

    tokens = _tokenize(qn)
    combined["_score"] = combined.apply(lambda row: _row_search_score(row, tokens, date_iso, matched_course), axis=1)

    if wants_highest and wants_occupancy and "Occupancy" in combined.columns:
        ranked = combined[pd.notna(combined["Occupancy"])].copy()
        if ranked.empty:
            return "No matching rows found in the dataset. Try a different course name, date, or keyword."
        ranked = ranked.sort_values(["Occupancy", "_score"], ascending=[False, False])
    elif wants_lowest and wants_occupancy and "Occupancy" in combined.columns:
        ranked = combined[pd.notna(combined["Occupancy"])].copy()
        if ranked.empty:
            return "No matching rows found in the dataset. Try a different course name, date, or keyword."
        ranked = ranked.sort_values(["Occupancy", "_score"], ascending=[True, False])
    elif wants_highest and wants_price and "Price" in combined.columns:
        ranked = combined[pd.notna(combined["Price"])].copy()
        if ranked.empty:
            return "No matching rows found in the dataset. Try a different course name, date, or keyword."
        ranked = ranked.sort_values(["Price", "_score"], ascending=[False, False])
    elif wants_lowest and wants_price and "Price" in combined.columns:
        ranked = combined[pd.notna(combined["Price"])].copy()
        if ranked.empty:
            return "No matching rows found in the dataset. Try a different course name, date, or keyword."
        ranked = ranked.sort_values(["Price", "_score"], ascending=[True, False])
    else:
        ranked = combined.sort_values(["_score", "Date", "Course Name"], ascending=[False, True, True])

    if not asks_data and not matched_course and not date_iso and not wants_price and not wants_occupancy and not wants_availability and not wants_market:
        ranked = ranked.sort_values(["_score", "Course Name", "Date"], ascending=[False, True, True])

    columns = _infer_columns(qn, ranked)
    if parsed_date and asks_data:
        # For date queries: render market and availability rows separately so each
        # table only shows columns that actually have data — no blank columns.
        market_rows = ranked[ranked["Source"] == "market_rates"].copy()
        drive_rows  = ranked[ranked["Source"] == "drive_dataset"].copy()
        av_rows     = ranked[ranked["Source"] == "availability"].copy()

        # Merge market + drive rows (same schema)
        market_combined = pd.concat([market_rows, drive_rows], ignore_index=True) if not drive_rows.empty else market_rows

        def _non_blank_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
            """Return only columns from candidates that exist and have at least one non-blank value."""
            result = []
            for col in candidates:
                if col in df.columns:
                    non_blank = df[col].replace("", pd.NA).dropna()
                    if not non_blank.empty:
                        result.append(col)
            return result

        market_cols = _non_blank_columns(market_combined, [
            "Course Name", "Date", "Price", "Occupancy",
            "Market Avg", "Market Min", "Market Max", "Source",
        ])
        av_cols = _non_blank_columns(av_rows, [
            "Course Name", "Date", "Time",
            "Brand Price", "GolfNow Price", "TeeOff Price", "SupremeGolf Price",
            "Brand Status", "GolfNow Status", "TeeOff Status", "SupremeGolf Status",
            "Availability", "Source",
        ])

        parts = []
        if not market_combined.empty and market_cols:
            parts.append(f"<h3>Market Rates — {date_iso}</h3>" + _table_from_frame(market_combined, market_cols, deduplicate=True))
        if not av_rows.empty and av_cols:
            parts.append(f"<h3>Availability — {date_iso}</h3>" + _table_from_frame(av_rows, av_cols, deduplicate=True))

        if parts:
            return "\n".join(parts)
        # Fall through to generic render if no source-split data found

    if not columns:
        columns = [column for column in ["Course Name", "Date", "Price", "Occupancy", "Availability", "Source"] if column in ranked.columns]

    if not columns:
        return "No matching rows found in the dataset. Try a different course name, date, or keyword."

    row_limit = None if parsed_date else 10
    return _table_from_frame(ranked, columns, limit=row_limit, deduplicate=not bool(parsed_date))
