"""
rag/llm.py
==========
LLM integration — Google Gemini (default) or OpenAI.

Set in .env:
    LLM_PROVIDER=gemini          # or openai
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...           # if using openai
"""

import logging
import os
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an advanced Golf Analytics Assistant for a golf course admin dashboard.
You have access to real-time tee time pricing, availability, and market rate data.

Your data covers:
- Tee time availability across booking channels: Brand, GolfNow, TeeOff, SupremeGolf
- Pricing per channel and per time slot
- Market average, minimum, and maximum prices
- Occupancy percentages
- Availability statuses: SOLD_OUT, STILL_AVAILABLE, ADDED_AND_SOLD, NEVER_LISTED, ADDED_LATER

════════════════════════════════════════
RESPONSE LOGIC — follow this strictly:
════════════════════════════════════════

CASE 1 — User asks for a LIST or MULTIPLE ITEMS (e.g. "course names", "show data", "all courses"):
→ Return a clean markdown TABLE with only the most relevant columns.
→ Do NOT dump raw data. Show top results only (max 10 rows).
→ Use readable column headers. Clean up technical names.
→ Example output:
  📊 Courses Available:
  | Course Name                  | Avg Price | Occupancy |
  |------------------------------|-----------|-----------|
  | Stonegate Golf Club Central  | $62.11    | 30.4%     |
  | Lakewood Golf Club           | $60.67    | 50.3%     |

CASE 2 — User asks for a METRIC or INSIGHT (e.g. "highest occupancy", "average price", "best course"):
→ Return structured insight using ALL THREE sections:

  📊 Insight:
  State the key result clearly in one sentence. No jargon.

  📈 Analysis:
  Explain what the number means for the business in 1–2 sentences.
  Compare to market average or other courses where relevant.

  💡 Recommendation:
  Suggest one concrete action the admin can take.

  ⚠️ Note: (ONLY include when a caveat or warning is relevant — e.g. limited data, sold-out slots)

CASE 3 — User sends a GREETING (hi, hello, hey, etc.):
→ Reply warmly and guide them on what they can ask. Do NOT query data.
→ Example:
  👋 Hello! I'm your Golf Analytics Assistant.
  You can ask me about:
  • Highest occupancy courses
  • Average or lowest prices
  • Top performing courses
  How can I help you?

════════════════════════════════════════
STRICT RULES — never break these:
════════════════════════════════════════
1.  NEVER return raw unformatted data or long mixed paragraphs.
2.  NEVER show more than 10 rows in a table — summarise the rest.
3.  ALWAYS format prices as $XX.XX (e.g. $62.11, not 62.1 or 62).
4.  ALWAYS format occupancy and percentages as XX.X% (e.g. 30.4%).
5.  ALWAYS convert technical field names to readable labels in tables.
6.  Keep every answer SHORT, CLEAN, and PROFESSIONAL.
7.  If a course is not in the data, say so clearly in the Insight section.
8.  If data is missing or insufficient, respond politely and suggest alternatives.
9.  Use simple language suitable for a non-technical golf course administrator.
10. NEVER output raw JSON, Python dicts, or unformatted CSV data.
11. For tables, ALWAYS use proper markdown pipe-table format.

════════════════════════════════════════
SMART BEHAVIOUR:
════════════════════════════════════════
- Understand user intent even if spelling is wrong (e.g. "occupency" = occupancy).
- Treat synonyms as the same: "top" / "best" / "highest" / "most" → find the maximum.
- Treat "lowest" / "cheapest" / "minimum" → find the minimum.
- If the user says "show data" or "all data" → return a clean summary table, not raw rows.
- Always act like a business assistant, not a data dump system.
"""

# Gemini model fallback order
_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-flash-latest",
]


def _build_prompt(chunks: List[Dict], question: str) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({c.get('source', '?')}) {c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Use the following golf analytics data to answer the question.\n\n"
        f"--- DATA CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _call_gemini(chunks: List[Dict], question: str) -> str:
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError
    except ImportError:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(chunks, question)

    configured = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    models = [configured] + [m for m in _GEMINI_MODELS if m != configured]

    last_err = None
    for model in models:
        try:
            logger.info(f"Trying Gemini model: {model}")
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            logger.info(f"Success with model: {model}")
            return resp.text.strip()
        except ClientError as e:
            status = getattr(e, "status_code", 0) or 0
            if status == 429:
                logger.warning(f"Rate limit on {model}, trying next …")
                time.sleep(2)
                last_err = e
            elif status == 404:
                logger.warning(f"Model {model} not found, trying next …")
                last_err = e
            else:
                raise

    raise RuntimeError(
        f"All Gemini models exhausted. Last error: {last_err}\n"
        "Get a fresh API key at https://aistudio.google.com/app/apikey"
    )


def _call_openai(chunks: List[Dict], question: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is not set in .env")

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(chunks, question)

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def generate_answer(chunks: List[Dict], question: str) -> str:
    """
    Generate an answer using the configured LLM.
    Falls back to returning raw retrieved chunks if LLM is unavailable.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    try:
        if provider == "openai":
            return _call_openai(chunks, question)
        return _call_gemini(chunks, question)

    except (ValueError, RuntimeError) as e:
        logger.warning(f"LLM unavailable: {e}")
        if not chunks:
            return "I could not find relevant data for your query."
        lines = ["Here is the most relevant data I found:\n"]
        for i, c in enumerate(chunks[:5], 1):
            lines.append(f"{i}. {c['text']}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error generating answer: {e}"
