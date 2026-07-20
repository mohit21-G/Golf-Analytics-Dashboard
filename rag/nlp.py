"""
rag/nlp.py
==========
NLP preprocessing layer for the Golf Analytics RAG bot.

Provides:
  - Stopword removal
  - Simple stemming (suffix stripping)
  - Golf-domain synonym expansion
  - Query intent detection (list vs metric vs comparison)
  - Query normalisation before retrieval
"""

import re
from typing import List, Tuple

# ── Golf-domain stopwords (words that add noise to retrieval) ─────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "neither", "not", "only", "own", "same",
    "than", "too", "very", "just", "me", "my", "myself", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "what", "which",
    "who", "this", "that", "these", "those", "i", "am", "tell", "show",
    "give", "get", "find", "want", "know", "see", "look", "please",
    "can", "could", "would", "how", "much", "many", "any", "all",
}

# ── Golf-domain synonym map ────────────────────────────────────────────────────
# Maps user words → canonical index terms used in the chunks
_SYNONYMS = {
    # Price synonyms
    "price":        "price",
    "pricing":      "price",
    "cost":         "price",
    "rate":         "price",
    "rates":        "price",
    "fee":          "price",
    "fees":         "price",
    "charge":       "price",
    "charges":      "price",
    "amount":       "price",
    "fare":         "price",
    "tariff":       "price",
    "prise":        "price",   # common misspelling
    "prce":         "price",   # common misspelling

    # Occupancy synonyms
    "occupancy":    "occupancy",
    "occupency":    "occupancy",  # misspelling
    "ocupancy":     "occupancy",  # misspelling
    "occupansy":    "occupancy",  # misspelling
    "demand":       "occupancy",
    "utilization":  "occupancy",
    "utilisation":  "occupancy",
    "booked":       "occupancy",
    "booking":      "occupancy",
    "bookings":     "occupancy",
    "filled":       "occupancy",
    "capacity":     "occupancy",

    # Availability synonyms
    "availability": "availability",
    "availabilty":  "availability",  # misspelling
    "availibility": "availability",  # misspelling
    "available":    "availability",
    "open":         "availability",
    "slot":         "availability",
    "slots":        "availability",
    "tee":          "tee",
    "teetime":      "tee",
    "tee-time":     "tee",

    # Course synonyms
    "course":       "course",
    "courses":      "course",
    "club":         "course",
    "clubs":        "course",
    "golf":         "golf",
    "venue":        "course",
    "venues":       "course",
    "location":     "course",

    # Channel synonyms
    "golfnow":      "golfnow",
    "golf now":     "golfnow",
    "teeoff":       "teeoff",
    "tee off":      "teeoff",
    "supremegolf":  "supremegolf",
    "supreme golf": "supremegolf",
    "brand":        "brand",
    "direct":       "brand",
    "channel":      "channel",
    "channels":     "channel",
    "platform":     "channel",

    # Market synonyms
    "market":       "market",
    "competitor":   "market",
    "competitors":  "market",
    "benchmark":    "market",
    "industry":     "market",
    "average":      "average",
    "avg":          "average",
    "mean":         "average",

    # Superlatives — map to search-friendly terms
    "highest":      "highest",
    "highest":      "highest",
    "top":          "highest",
    "best":         "highest",
    "most":         "highest",
    "maximum":      "highest",
    "max":          "highest",
    "peak":         "highest",
    "lowest":       "lowest",
    "cheapest":     "lowest",
    "minimum":      "lowest",
    "min":          "lowest",
    "least":        "lowest",
    "bottom":       "lowest",
    "worst":        "lowest",

    # Status synonyms
    "sold":         "sold_out",
    "soldout":      "sold_out",
    "sold out":     "sold_out",
    "full":         "sold_out",
    "unavailable":  "sold_out",
    "still available": "still_available",
    "open":         "still_available",
}

# ── Intent patterns ───────────────────────────────────────────────────────────
# Used to detect what kind of response the user expects

_LIST_PATTERNS = [
    r"\blist\b", r"\ball\b", r"\bnames?\b", r"\bshow\b", r"\bdisplay\b",
    r"\bwhat courses?\b", r"\bwhich courses?\b", r"\bname of\b",
    r"\bshow data\b", r"\ball data\b", r"\bshow me\b",
]

_METRIC_PATTERNS = [
    r"\bhighest\b", r"\blowest\b", r"\bbest\b", r"\bworst\b", r"\btop\b",
    r"\baverage\b", r"\bavg\b", r"\btotal\b", r"\bsum\b", r"\bcount\b",
    r"\bcompare\b", r"\bcomparison\b", r"\bvs\b", r"\bversus\b",
    r"\btrend\b", r"\bperformance\b", r"\binsight\b", r"\banalysis\b",
    r"\bhow much\b", r"\bwhat is the\b", r"\bwhat are the\b",
]


def detect_intent(query: str) -> str:
    """
    Detect the user's intent from the query.

    Returns:
        "list"       — user wants a table of items
        "metric"     — user wants a specific insight/number
        "general"    — general question
    """
    q = query.lower()
    for pattern in _LIST_PATTERNS:
        if re.search(pattern, q):
            return "list"
    for pattern in _METRIC_PATTERNS:
        if re.search(pattern, q):
            return "metric"
    return "general"


def _stem(word: str) -> str:
    """
    Very lightweight suffix stemmer for English.
    Handles common suffixes without requiring NLTK.
    """
    if len(word) <= 4:
        return word
    for suffix in ("ing", "tion", "tions", "ness", "ment", "ments",
                   "ies", "ied", "ers", "er", "est", "ly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess(text: str, stem: bool = False) -> List[str]:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Tokenise (alphanumeric + decimals)
      3. Remove stopwords
      4. Apply synonym expansion
      5. Optional stemming
      6. Add bigrams

    Returns a list of processed tokens.
    """
    text = text.lower().strip()

    # Multi-word synonym replacement before tokenising
    for phrase, replacement in _SYNONYMS.items():
        if " " in phrase:
            text = text.replace(phrase, replacement)

    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)

    processed = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        # Single-word synonym expansion
        tok = _SYNONYMS.get(tok, tok)
        if stem:
            tok = _stem(tok)
        processed.append(tok)

    # Add bigrams from processed tokens
    bigrams = [
        f"{processed[i]}_{processed[i+1]}"
        for i in range(len(processed) - 1)
    ]

    return processed + bigrams


def expand_query(query: str) -> str:
    """
    Expand a user query with synonyms and related terms to improve retrieval.

    Example:
        "cheapest course" → "cheapest course lowest price minimum avg_price"
    """
    q_lower = query.lower()
    expansions = []

    # Price intent
    if any(w in q_lower for w in ["price", "cost", "rate", "fee", "cheap", "expensive", "prise"]):
        expansions += ["price", "avg_price", "market_avg"]

    # Occupancy intent
    if any(w in q_lower for w in ["occupancy", "demand", "booked", "occupency", "ocupancy"]):
        expansions += ["occupancy", "occ_percent"]

    # Availability intent
    if any(w in q_lower for w in ["available", "availability", "slot", "open", "tee"]):
        expansions += ["availability", "still_available", "tee"]

    # Superlative intent
    if any(w in q_lower for w in ["highest", "top", "best", "most", "maximum", "max"]):
        expansions += ["highest", "maximum", "summary"]

    if any(w in q_lower for w in ["lowest", "cheapest", "minimum", "min", "least"]):
        expansions += ["lowest", "minimum", "summary"]

    # Market comparison intent
    if any(w in q_lower for w in ["market", "compare", "vs", "versus", "benchmark"]):
        expansions += ["market_avg", "market_min", "market_max", "summary"]

    if expansions:
        return query + " " + " ".join(set(expansions))
    return query


def normalise_query(query: str) -> Tuple[str, str]:
    """
    Full query normalisation pipeline.

    Returns:
        (expanded_query, intent)
        - expanded_query: query enriched with synonyms and related terms
        - intent: "list", "metric", or "general"
    """
    intent = detect_intent(query)
    expanded = expand_query(query)
    return expanded, intent
