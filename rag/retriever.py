"""
rag/retriever.py
================
BM25 retriever with NLP query expansion.

Pipeline:
  1. Expand query using NLP synonym + intent layer (rag/nlp.py)
  2. Build BM25 query vector
  3. FAISS cosine similarity search → top-k chunks
  4. Filter out low-confidence results
"""

import logging
import math
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss

from rag.ingest import INDEX_FILE, CHUNKS_FILE, VOCAB_FILE, build_index, _tokenize

logger = logging.getLogger(__name__)

# Minimum similarity score to include a result (0.0 = include all)
_MIN_SCORE = 0.01

# Module-level singletons (loaded once)
_index:  faiss.Index | None = None
_chunks: List[Dict[str, Any]] | None = None
_vocab:  Dict[str, int] | None = None
_idf:    np.ndarray | None = None


def _load() -> None:
    global _index, _chunks, _vocab, _idf

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists() or not VOCAB_FILE.exists():
        logger.info("Index not found — building now …")
        build_index()

    logger.info("Loading FAISS index …")
    _index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as f:
        _chunks = pickle.load(f)

    with open(VOCAB_FILE, "rb") as f:
        data   = pickle.load(f)
        _vocab = data["vocab"]
        _idf   = data["idf"]

    logger.info(f"Retriever ready — {_index.ntotal} chunks, vocab={len(_vocab)}")


def _query_vector(query: str) -> np.ndarray:
    """Build a normalised BM25 query vector from the query string."""
    from collections import Counter
    toks = _tokenize(query)
    tf   = Counter(toks)
    V    = len(_vocab)
    vec  = np.zeros(V, dtype="float32")
    for tok, cnt in tf.items():
        if tok in _vocab:
            vec[_vocab[tok]] = (1 + math.log(cnt)) * _idf[_vocab[tok]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.reshape(1, -1)


def retrieve(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Pipeline:
      1. Expand query with NLP synonyms + intent signals
      2. Build BM25 query vector
      3. FAISS cosine similarity search
      4. Filter low-confidence results
    """
    global _index, _chunks, _vocab, _idf

    if _index is None:
        _load()

    # Step 1: NLP query expansion
    try:
        from rag.nlp import normalise_query
        expanded_query, intent = normalise_query(query)
        logger.info(f"Query: {query!r} → expanded: {expanded_query!r} | intent: {intent}")
    except ImportError:
        expanded_query = query
        intent = "general"

    # Step 2: Build query vector from expanded query
    q_vec = _query_vector(expanded_query)

    # Step 3: FAISS search (fetch extra candidates for filtering)
    fetch_k = min(top_k * 3, _index.ntotal)
    scores, indices = _index.search(q_vec, fetch_k)

    # Step 4: Filter and rank results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < _MIN_SCORE:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"]  = round(float(score), 4)
        chunk["intent"] = intent
        results.append(chunk)

    # Return top_k after filtering
    return results[:top_k]


def add_chunks(new_chunks: List[Dict[str, Any]]) -> None:
    """Add new chunks to the in-memory index (does not persist to disk)."""
    global _index, _chunks, _vocab, _idf

    if _index is None:
        _load()

    from collections import Counter
    V    = len(_vocab)
    vecs = []
    for c in new_chunks:
        toks = _tokenize(c["text"])
        tf   = Counter(toks)
        vec  = np.zeros(V, dtype="float32")
        for tok, cnt in tf.items():
            if tok in _vocab:
                vec[_vocab[tok]] = (1 + math.log(cnt)) * _idf[_vocab[tok]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vecs.append(vec)

    matrix = np.array(vecs, dtype="float32")
    _index.add(matrix)
    _chunks.extend(new_chunks)
    logger.info(f"Added {len(new_chunks)} chunks. Total: {_index.ntotal}")
