"""
rag/ingest.py
=============
Builds a BM25-weighted TF-IDF + FAISS index from the two CSV files.
Uses the NLP preprocessing layer (rag/nlp.py) for better tokenisation,
synonym expansion, and stopword removal.

Usage:
    python -m rag.ingest          # build once
    python -m rag.ingest --force  # force rebuild
"""

import logging
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss

logger = logging.getLogger(__name__)

# BM25 hyperparameters
_BM25_K1 = 1.5   # term frequency saturation (1.2–2.0 typical)
_BM25_B  = 0.75  # length normalisation (0 = no normalisation, 1 = full)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
AVAILABILITY_CSV = BASE_DIR / "Availability.csv"
MARKET_RATES_CSV = BASE_DIR / "Market Rates.csv"
INDEX_DIR        = BASE_DIR / "rag" / "index"
INDEX_FILE       = INDEX_DIR / "golf.faiss"
CHUNKS_FILE      = INDEX_DIR / "chunks.pkl"
VOCAB_FILE       = INDEX_DIR / "vocab.pkl"

# ── Google Drive dataset (same source as project.py) ──────────────────────────
_DRIVE_FILE_ID   = "1u0-AbwITOPix3_x-8wWTrWCI-IMl6tIz"
_DRIVE_CSV_CACHE = INDEX_DIR / "drive_dataset.csv"


def _drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _load_drive_dataset() -> pd.DataFrame:
    """
    Download the Google Drive dataset used by project.py and cache it locally.
    Returns an empty DataFrame (with a warning) if the download fails so the
    rest of the pipeline is never blocked.
    """
    # Use cached copy if available
    if _DRIVE_CSV_CACHE.exists():
        logger.info(f"Loading cached Drive dataset from {_DRIVE_CSV_CACHE}")
        try:
            df = pd.read_csv(_DRIVE_CSV_CACHE)
            df.columns = df.columns.str.strip().str.lower()
            logger.info(f"Drive dataset loaded from cache: {len(df)} rows")
            return df
        except Exception as exc:
            logger.warning(f"Cache read failed ({exc}), re-downloading …")

    # Download fresh copy
    url = _drive_download_url(_DRIVE_FILE_ID)
    logger.info(f"Downloading Drive dataset from {url} …")
    try:
        import requests
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        _DRIVE_CSV_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _DRIVE_CSV_CACHE.write_bytes(resp.content)
        logger.info(f"Drive dataset downloaded and cached at {_DRIVE_CSV_CACHE}")
        df = pd.read_csv(_DRIVE_CSV_CACHE)
        df.columns = df.columns.str.strip().str.lower()
        logger.info(f"Drive dataset rows: {len(df)}")
        return df
    except Exception as exc:
        logger.warning(
            f"Could not download Drive dataset ({exc}). "
            "Continuing without it — only local CSVs will be indexed."
        )
        return pd.DataFrame()


# ── tokeniser (delegates to NLP layer) ────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Tokenise text using the NLP preprocessing pipeline.
    Falls back to basic tokenisation if nlp module unavailable.
    """
    try:
        from rag.nlp import preprocess
        return preprocess(text, stem=False)
    except ImportError:
        # Fallback: basic tokenisation + bigrams
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams


# ── row → text ─────────────────────────────────────────────────────────────────

def _availability_row_to_text(row: pd.Series) -> str:
    parts = [
        f"Course {row.get('course_name', 'N/A')}",
        f"Date {row.get('tee_date', 'N/A')}",
        f"Time {row.get('tee_time', 'N/A')}",
    ]
    for ch in ("brand", "golfnow", "teeoff", "supremegolf"):
        price  = row.get(f"{ch}_current_price")
        status = row.get(f"{ch}_availability_status", "N/A")
        if pd.notna(price):
            parts.append(f"{ch} price {float(price):.2f} status {status}")
        else:
            parts.append(f"{ch} status {status}")
    parts.append(f"overall availability {row.get('overall_availability_status', 'N/A')}")
    return " ".join(parts)


def _market_row_to_text(row: pd.Series) -> str:
    return (
        f"course {row.get('course_name', 'N/A')} "
        f"as of {row.get('as_of_date', 'N/A')} "
        f"tee date {row.get('tee_date', 'N/A')} "
        f"avg price {float(row['avg_price']):.2f} "
        f"occupancy {float(row['occ_percent']):.1f} percent "
        f"market avg {float(row['market_avg']):.2f} "
        f"market min {float(row['market_min']):.2f} "
        f"market max {float(row['market_max']):.2f}"
    )


def _build_summary_chunks(market_df: pd.DataFrame) -> List[Dict[str, Any]]:
    chunks = []
    for course, grp in market_df.groupby("course_name"):
        text = (
            f"summary {course} "
            f"average price {grp['avg_price'].mean():.2f} "
            f"average occupancy {grp['occ_percent'].mean():.1f} percent "
            f"average market price {grp['market_avg'].mean():.2f} "
            f"data from {grp['tee_date'].min()} to {grp['tee_date'].max()}"
        )
        chunks.append({"text": text, "source": "market_summary", "course": course})

    for (course, tee_date), grp in market_df.groupby(["course_name", "tee_date"]):
        r = grp.iloc[0]
        text = (
            f"on {tee_date} course {course} "
            f"avg price {r['avg_price']:.2f} "
            f"occupancy {r['occ_percent']:.1f} percent "
            f"market avg {r['market_avg']:.2f} "
            f"market min {r['market_min']:.2f} "
            f"market max {r['market_max']:.2f}"
        )
        chunks.append({
            "text": text, "source": "market_daily",
            "course": course, "date": str(tee_date),
        })
    return chunks


# ── BM25-weighted TF-IDF ──────────────────────────────────────────────────────

def _build_tfidf(
    texts: List[str],
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """
    Build a BM25-weighted document-term matrix.

    BM25 improves on plain TF-IDF by:
    - Saturating term frequency (very frequent terms don't dominate)
    - Normalising by document length (short chunks aren't penalised)
    """
    n = len(texts)
    tokenized = [_tokenize(t) for t in texts]

    # Document frequency count
    df_count: Counter = Counter()
    for toks in tokenized:
        df_count.update(set(toks))

    # Keep tokens that appear in at least 2 documents
    filtered = [(tok, cnt) for tok, cnt in sorted(df_count.items()) if cnt >= 2]
    vocab = {tok: i for i, (tok, _) in enumerate(filtered)}
    V = len(vocab)
    logger.info(f"Vocabulary size: {V}")

    # IDF with smoothing (same as before — BM25 IDF variant)
    idf = np.zeros(V, dtype="float32")
    for tok, idx in vocab.items():
        df = df_count[tok]
        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf[idx] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # Average document length for BM25 length normalisation
    doc_lengths = [len(toks) for toks in tokenized]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)

    # Build BM25 matrix
    matrix = np.zeros((n, V), dtype="float32")
    for doc_i, toks in enumerate(tokenized):
        tf: Counter = Counter(toks)
        dl = doc_lengths[doc_i]
        for tok, cnt in tf.items():
            if tok in vocab:
                # BM25 TF: (cnt * (k1 + 1)) / (cnt + k1 * (1 - b + b * dl/avg_dl))
                bm25_tf = (cnt * (_BM25_K1 + 1)) / (
                    cnt + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
                )
                matrix[doc_i, vocab[tok]] = bm25_tf * idf[vocab[tok]]

    # L2 normalise for cosine similarity in FAISS
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms

    return vocab, idf, matrix


# ── main ───────────────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    if (
        not force
        and INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
        and VOCAB_FILE.exists()
    ):
        logger.info("Index already exists. Use force=True to rebuild.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading CSV files …")

    avail_df = pd.read_csv(AVAILABILITY_CSV)
    avail_df.columns = avail_df.columns.str.strip().str.lower()

    market_df = pd.read_csv(MARKET_RATES_CSV)
    market_df.columns = market_df.columns.str.strip().str.lower()

    # Normalise tee_date to YYYY-MM-DD (strip any time component)
    for df in (avail_df, market_df):
        if "tee_date" in df.columns:
            df["tee_date"] = (
                pd.to_datetime(df["tee_date"], errors="coerce").dt.date
            )

    logger.info(f"Availability rows: {len(avail_df)}, Market rows: {len(market_df)}")

    # ── Load Google Drive dataset (same source as project.py) ─────────────────
    drive_df = _load_drive_dataset()
    if not drive_df.empty and "tee_date" in drive_df.columns:
        drive_df["tee_date"] = (
            pd.to_datetime(drive_df["tee_date"], errors="coerce").dt.date
        )
        logger.info(f"Drive dataset rows: {len(drive_df)}")
    else:
        if not drive_df.empty:
            logger.warning("Drive dataset has no 'tee_date' column — skipping date normalisation")
        drive_df = pd.DataFrame()

    chunks: List[Dict[str, Any]] = []

    for _, row in avail_df.iterrows():
        chunks.append({
            "text":   _availability_row_to_text(row),
            "source": "availability",
            "course": str(row.get("course_name", "")),
            "date":   str(row.get("tee_date", "")),
            "time":   str(row.get("tee_time", "")),
        })

    for _, row in market_df.iterrows():
        chunks.append({
            "text":   _market_row_to_text(row),
            "source": "market_rates",
            "course": str(row.get("course_name", "")),
            "date":   str(row.get("tee_date", "")),
        })

    chunks.extend(_build_summary_chunks(market_df))

    # ── Index Drive dataset rows ───────────────────────────────────────────────
    if not drive_df.empty:
        drive_chunks_added = 0
        for _, row in drive_df.iterrows():
            text = _market_row_to_text(row)
            chunks.append({
                "text":   text,
                "source": "drive_dataset",
                "course": str(row.get("course_name", "")),
                "date":   str(row.get("tee_date", "")),
            })
            drive_chunks_added += 1
        # Add per-course summaries for the drive dataset too
        if "avg_price" in drive_df.columns and "occ_percent" in drive_df.columns:
            chunks.extend(_build_summary_chunks(drive_df))
        logger.info(f"Drive dataset chunks added: {drive_chunks_added}")

    logger.info(
        f"Total chunks: {len(chunks)} "
        f"(availability={len(avail_df)}, market={len(market_df)}, "
        f"drive={len(drive_df) if not drive_df.empty else 0})"
    )

    texts = [c["text"] for c in chunks]
    logger.info("Building TF-IDF matrix …")
    vocab, idf, matrix = _build_tfidf(texts)

    dim   = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump({"vocab": vocab, "idf": idf}, f)

    logger.info(f"Index saved — {index.ntotal} vectors, dim={dim}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force = "--force" in sys.argv
    build_index(force=force)
