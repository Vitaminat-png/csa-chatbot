"""
api/retrieval.py
----------------
Query Pinecone, rerank by score, and return top-K chunks + relevant URL mappings.

Reranking pipeline (when RERANK_ENABLED=True):
  1. Fetch RERANK_TOP_K chunks from Pinecone (both namespaces)
  2. Score each chunk with GPT-4o-mini in parallel (asyncio.gather)
  3. Sort by GPT score, take FINAL_TOP_K for context
  4. Fallback to Pinecone cosine score if GPT call times out or fails
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone

from api.models import Source

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "csa-chatbot")
EMBED_MODEL = "text-embedding-3-small"
TOP_K = int(os.environ.get("TOP_K", 5))
MIN_SCORE = 0.30  # discard very-low-relevance chunks

# Reranking config
RERANK_ENABLED: bool = os.environ.get("RERANK_ENABLED", "true").lower() != "false"
RERANK_TOP_K: int = int(os.environ.get("RERANK_TOP_K", 10))   # fetch this many from Pinecone
FINAL_TOP_K: int = int(os.environ.get("FINAL_TOP_K", 4))      # keep this many after rerank
RERANK_TIMEOUT: float = 5.0   # seconds per GPT scoring call
RERANK_CACHE_TTL: float = 300.0  # cache rerank scores for 5 minutes

# ---------------------------------------------------------------------------
# URL blocklist — paths that are never useful to suggest to users
# ---------------------------------------------------------------------------
BLOCKED_URL_PATTERNS: list[str] = [
    "/shop/",
    "/cart/",
    "/checkout/",
    "/my-account/",
    "/carrello/",
    "/cassa/",
    "/mon-compte/",
    "/mi-cuenta/",
    "/boutique/",
    "/panier/",
]


def _is_blocked_url(url: str | None) -> bool:
    """Return True if *url* matches any blocked path pattern."""
    if not url:
        return False
    lower = url.lower()
    return any(pattern in lower for pattern in BLOCKED_URL_PATTERNS)


# ---------------------------------------------------------------------------
# Singletons (initialised lazily so import doesn't fail without keys)
# ---------------------------------------------------------------------------
_oai: Optional[OpenAI] = None
_async_oai: Optional[AsyncOpenAI] = None
_pc: Optional[Pinecone] = None
_index = None


def _get_clients() -> tuple[OpenAI, object]:
    global _oai, _pc, _index
    if _oai is None:
        _oai = OpenAI(api_key=OPENAI_API_KEY)
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX_NAME)
    return _oai, _index


def _get_async_oai() -> AsyncOpenAI:
    global _async_oai
    if _async_oai is None:
        _async_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _async_oai


# ---------------------------------------------------------------------------
# Rerank cache: key -> (gpt_score: float, timestamp: float)
# ---------------------------------------------------------------------------
_rerank_cache: dict[str, tuple[float, float]] = {}


async def _score_chunk(query: str, text: str, pinecone_score: float) -> float:
    """
    Ask GPT-4o-mini to score the relevance of *text* for *query* (0-10).
    Returns -1.0 as sentinel on error; caller falls back to pinecone_score.
    Uses a simple in-memory TTL cache keyed on (query, text).
    """
    cache_key = f"{hash(query)}:{hash(text)}"
    now = time.monotonic()
    cached = _rerank_cache.get(cache_key)
    if cached is not None:
        gpt_score, ts = cached
        if now - ts < RERANK_CACHE_TTL:
            return gpt_score

    prompt = (
        f"Valuta da 0 a 10 quanto questo testo è rilevante per rispondere alla domanda: "
        f"{query}\nTesto: {text}\nRispondi SOLO con il numero."
    )
    try:
        oai = _get_async_oai()
        resp = await asyncio.wait_for(
            oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            ),
            timeout=RERANK_TIMEOUT,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Accept both "7" and "7.5"
        gpt_score = float(raw.replace(",", "."))
        gpt_score = max(0.0, min(10.0, gpt_score))
    except asyncio.TimeoutError:
        logger.warning("rerank: GPT scoring timed out for chunk (query='%s...')", query[:40])
        return -1.0
    except (ValueError, Exception) as exc:
        logger.warning("rerank: GPT scoring failed: %s", exc)
        return -1.0

    _rerank_cache[cache_key] = (gpt_score, now)
    return gpt_score


async def rerank_chunks(query: str, chunks: list[tuple[str, float]]) -> list[tuple[float, int]]:
    """
    Rerank *chunks* using GPT-4o-mini in parallel.

    Parameters
    ----------
    query  : user query string
    chunks : list of (text, pinecone_score) tuples

    Returns
    -------
    List of (final_score, original_index) sorted descending by final_score.
    """
    tasks = [_score_chunk(query, text, pscore) for text, pscore in chunks]
    gpt_scores: list[float] = await asyncio.gather(*tasks)  # type: ignore[assignment]

    ranked: list[tuple[float, int]] = []
    for idx, (gpt_score, (_, pinecone_score)) in enumerate(zip(gpt_scores, chunks)):
        if gpt_score < 0:
            # Fallback: normalise Pinecone cosine score (0-1) to 0-10
            final_score = pinecone_score * 10.0
        else:
            final_score = gpt_score
        ranked.append((final_score, idx))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Language detection (lightweight heuristic; good enough for 4 languages)
# ---------------------------------------------------------------------------
_LANG_HINTS: dict[str, list[str]] = {
    "it": ["il", "la", "le", "che", "per", "con", "del", "della", "un", "una", "come", "questo"],
    "fr": ["le", "la", "les", "du", "des", "une", "pour", "avec", "comment", "est", "que"],
    "es": ["el", "la", "los", "las", "del", "para", "con", "cómo", "qué", "una", "como", "que"],
    "en": ["the", "is", "are", "how", "what", "does", "can", "for", "with", "this", "that"],
}


def detect_language(text: str) -> str:
    """
    Heuristic language detection based on common function words.
    Returns BCP-47 code: 'it' | 'en' | 'fr' | 'es' (default 'en').
    """
    lower = text.lower()
    words = set(lower.split())
    scores = {
        lang: sum(1 for w in hints if w in words)
        for lang, hints in _LANG_HINTS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "en"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    top_k: int = TOP_K,
    language_hint: Optional[str] = None,
) -> tuple[list[Source], str]:
    """
    Embed *query*, query Pinecone, optionally rerank with GPT-4o-mini,
    and return (sources, detected_language).

    Parameters
    ----------
    query          : user message text
    top_k          : number of chunks to return (ignored when RERANK_ENABLED;
                     FINAL_TOP_K is used instead)
    language_hint  : override language detection (e.g. from ChatRequest.language)

    Returns
    -------
    sources            : list[Source] sorted by score descending
    detected_language  : 'it' | 'en' | 'fr' | 'es'
    """
    detected_lang = language_hint or detect_language(query)
    oai, index = _get_clients()

    # Embed the query (sync SDK — fast, no benefit from async here)
    emb_response = oai.embeddings.create(model=EMBED_MODEL, input=[query])
    query_vector = emb_response.data[0].embedding

    # How many to fetch from Pinecone
    fetch_k = RERANK_TOP_K if RERANK_ENABLED else top_k
    _fetch_k = fetch_k * 3  # over-fetch to allow merge/dedup across namespaces

    raw_default = index.query(
        vector=query_vector,
        top_k=_fetch_k,
        include_metadata=True,
    )
    raw_catalog = index.query(
        vector=query_vector,
        top_k=_fetch_k,
        include_metadata=True,
        namespace="catalog",
    )

    seen_ids: set[str] = set()
    matches: list[dict] = []
    for m in raw_default.get("matches", []) + raw_catalog.get("matches", []):
        mid = m.get("id", "")
        if mid not in seen_ids:
            seen_ids.add(mid)
            matches.append(m)

    # Filter by MIN_SCORE and build intermediate list
    filtered: list[dict] = [m for m in matches if m.get("score", 0.0) >= MIN_SCORE]

    # Cap to RERANK_TOP_K (or top_k when reranking disabled) before scoring
    pre_rerank = filtered[:fetch_k]

    # -----------------------------------------------------------------------
    # Optional GPT reranking
    # -----------------------------------------------------------------------
    if RERANK_ENABLED and pre_rerank:
        try:
            # Use up to 500 chars of text for scoring (better than 200-char snippet)
            chunk_texts = [
                (m.get("metadata", {}).get("text", "")[:500], m.get("score", 0.0))
                for m in pre_rerank
            ]
            ranked = await rerank_chunks(query, chunk_texts)
            # Keep FINAL_TOP_K after rerank
            top_indices = [idx for _, idx in ranked[:FINAL_TOP_K]]
            selected_matches = [pre_rerank[i] for i in top_indices]
            logger.debug(
                "rerank: kept %d/%d chunks for query='%s...'",
                len(selected_matches), len(pre_rerank), query[:40],
            )
        except Exception as exc:
            logger.warning("rerank: pipeline failed (%s), falling back to Pinecone order", exc)
            selected_matches = pre_rerank[:FINAL_TOP_K]
    else:
        selected_matches = pre_rerank[:top_k]

    # -----------------------------------------------------------------------
    # Build Source objects
    # -----------------------------------------------------------------------
    sources: list[Source] = []
    for match in selected_matches:
        score: float = match.get("score", 0.0)
        meta: dict = match.get("metadata", {})

        # Pick the right language URL for url_mapping entries
        url: Optional[str] = None
        if meta.get("type") == "url_mapping":
            url = (
                meta.get(f"url_{detected_lang}")
                or meta.get("url_en")
                or meta.get("url_it")
                or ""
            )

        # Discard URLs that match blocked patterns (shop/cart/checkout/account pages)
        if _is_blocked_url(url):
            url = None

        sources.append(
            Source(
                source_file=meta.get("source_file", "unknown"),
                page=meta.get("page"),
                chunk_id=meta.get("chunk_id", match["id"]),
                score=round(score, 4),
                text_snippet=(meta.get("text", "")[:200]),
                url=url or None,
                product_family=meta.get("product_family") or None,
                valve_model=meta.get("valve_model") or None,
            )
        )

    # When reranking is disabled, enforce score-descending order
    if not RERANK_ENABLED:
        sources.sort(key=lambda s: s.score, reverse=True)

    return sources, detected_lang


# ---------------------------------------------------------------------------
# Context builder — converts sources to a formatted string for the prompt
# ---------------------------------------------------------------------------
def build_context_string(sources: list[Source], detected_lang: str) -> str:
    """
    Format retrieved sources into a context block for the system prompt.
    """
    if not sources:
        return ""

    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        lines.append(f"--- Source {i} (score={src.score}) ---")
        if src.source_file != "web_scraper":
            lines.append(f"File: {src.source_file}, Page: {src.page}")
        if src.url:
            lines.append(f"URL ({detected_lang}): {src.url}")
        lines.append(src.text_snippet)
        lines.append("")

    return "\n".join(lines)
