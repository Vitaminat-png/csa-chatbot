"""
api/retrieval.py
----------------
Query Pinecone, rerank by score, and return top-K chunks + relevant URL mappings.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from api.models import Source

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "csa-chatbot")
EMBED_MODEL = "text-embedding-3-small"
TOP_K = int(os.environ.get("TOP_K", 5))
MIN_SCORE = 0.30  # discard very-low-relevance chunks

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
_pc: Optional[Pinecone] = None
_index = None


def _get_clients():
    global _oai, _pc, _index
    if _oai is None:
        _oai = OpenAI(api_key=OPENAI_API_KEY)
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX_NAME)
    return _oai, _index


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
def retrieve(
    query: str,
    top_k: int = TOP_K,
    language_hint: Optional[str] = None,
) -> tuple[list[Source], str]:
    """
    Embed *query*, query Pinecone, rerank by score, return (sources, detected_language).

    Parameters
    ----------
    query          : user message text
    top_k          : number of chunks to return
    language_hint  : override language detection (e.g. from ChatRequest.language)

    Returns
    -------
    sources            : list[Source] sorted by score descending
    detected_language  : 'it' | 'en' | 'fr' | 'es'
    """
    detected_lang = language_hint or detect_language(query)
    oai, index = _get_clients()

    # Embed the query
    emb_response = oai.embeddings.create(model=EMBED_MODEL, input=[query])
    query_vector = emb_response.data[0].embedding

    # Query Pinecone — retrieve more than top_k to allow reranking
    raw = index.query(
        vector=query_vector,
        top_k=top_k * 3,  # over-fetch to rerank
        include_metadata=True,
    )

    matches = raw.get("matches", [])

    # Build Source objects; filter by MIN_SCORE
    sources: list[Source] = []
    for match in matches:
        score: float = match.get("score", 0.0)
        if score < MIN_SCORE:
            continue
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
            )
        )

    # Sort by score descending (Pinecone already does this, but enforce after filter)
    sources.sort(key=lambda s: s.score, reverse=True)
    sources = sources[:top_k]

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
