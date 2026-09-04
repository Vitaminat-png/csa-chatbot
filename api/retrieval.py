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
import json
import logging
import os
import re
import time
import unicodedata
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone

from api.model_index import (
    applications_for,
    find_application_sources,
    features_for,
    find_feature_sources,
    find_category_products,
    find_sizing_programs,
    find_exact_model_source,
    find_family,
    find_model_sources,
    series_on_page,
    find_sections,
    find_series_documents,
    files_with_sections,
    is_catalogue,
    section_of,
    sources_mentioned,
)
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
# Candidates sent to the reranker. Reranking is a single batched call, so a
# wider pool costs one longer prompt rather than more calls — and recall matters:
# with a pool of 12, near-duplicate chunks from one document filled every slot
# and pushed the actually-relevant page out before it could be scored.
RERANK_TOP_K: int = int(os.environ.get("RERANK_TOP_K", 20))
FINAL_TOP_K: int = int(os.environ.get("FINAL_TOP_K", 6))      # keep this many after rerank
# Seconds for the single batched scoring call. On timeout retrieval falls back
# to Pinecone order, which also bypasses the relevance floor — so the budget is
# set generously rather than tightly.
RERANK_TIMEOUT: float = 18.0
RERANK_CACHE_TTL: float = 300.0  # cache rerank scores for 5 minutes
RERANK_EXCERPT_CHARS: int = 600  # chars per chunk shown to the reranker

# Max characters of a single chunk passed into the LLM context. Chunks are ~500
# tokens (~2000 chars) by design, so this is a safety valve, not a routine cut.
MAX_CONTEXT_CHARS_PER_SOURCE: int = 2500

# How many chunks either side of a selected one are pulled in to complete it,
# and the larger cap that applies once they are attached. Kept deliberately
# small: padding the context is what made the model refuse to answer about the
# ITALICA 353 when four unrelated chunks were added alongside its datasheet.
NEIGHBOUR_SPAN: int = 1
MAX_CONTEXT_CHARS_WITH_NEIGHBOURS: int = 6000

# ---------------------------------------------------------------------------
# Language-aware scoring
# ---------------------------------------------------------------------------
# The corpus holds the same content in several languages (site pages and the XLC
# engineering PDFs exist in it/en/fr/es). Without this, an Italian question can
# retrieve the Spanish copy of the right page and waste a context slot.
LANG_MATCH_BOOST = 1.18      # chunk language == user language
LANG_ENGLISH_BOOST = 1.0     # English is the technical source of truth: neutral
LANG_MISMATCH_PENALTY = 0.55  # a different language: same content, wrong words

# The XLC engineering documents supersede the older per-model XLC datasheets in
# docs/, so their chunks outrank equally-similar legacy ones.
DOC_PRIORITY_BOOST = 1.12

# ---------------------------------------------------------------------------
# Exact model-code matching
# ---------------------------------------------------------------------------
MODEL_MATCH_TOP_K = 8       # chunks pulled from the named model's datasheet
MODEL_MATCH_BOOST = 1.6     # weight applied to chunks from that datasheet
# The datasheet carrying exactly the code the user typed, fetched on its own so
# its siblings cannot crowd it out, and weighted above them.
EXACT_MODEL_TOP_K = 6
EXACT_MODEL_BOOST = 2.0
# Context slots held for it before anything else is chosen.
EXACT_MODEL_RESERVED_SLOTS = 2
# Context slots held for chunks that demonstrably contain the column asked for.
ATTRIBUTE_PINNED_SLOTS = 2
# Context slots held for the named model's datasheet, so the reranker cannot
# crowd out the authoritative document with merely similar-looking chunks.
MODEL_RESERVED_SLOTS = 2

# Chunks pulled from the datasheets of a named application. Larger than the
# model equivalent because the point is breadth across products, and the
# per-source cap then keeps any one of them from taking over.
APPLICATION_MATCH_TOP_K = 20
# Datasheet chunks score 0.31-0.42 against an application question while the
# Italian catalogue scores 0.56, so without a lift they never reach the
# reranker. The reranker's relevance floor then discards whatever is not
# actually useful, so the lift costs nothing when it is unwarranted.
APPLICATION_MATCH_BOOST = 1.5
# On an application question the answer should span products, so no single
# datasheet may take more than this many of the candidate slots.
APPLICATION_MAX_CHUNKS_PER_SOURCE = 2
# Context slots held for distinct products documented for the application, and a
# wider context for these questions. "Do you have valves for drinking water?"
# is answered by a range, but the site's own application pages score higher than
# any single datasheet and took four of the six slots, leaving room for one
# product — an implausible answer from a waterworks manufacturer.
APPLICATION_RESERVED_SLOTS = 4
APPLICATION_FINAL_TOP_K = 8

# ---------------------------------------------------------------------------
# Cross-language query expansion
# ---------------------------------------------------------------------------
# The 116 product datasheets in docs/ are English-only, and a question asked in
# Italian scores them far below the Italian catalogue: "per irrigazione cosa
# consigli?" ranked ARGO.pdf 8th at 0.390, while "valves for irrigation" ranked
# it 1st at 0.543 — yet ARGO's datasheet opens by saying it is *for irrigation*.
# Searching with an English rendering of the question as well puts the English
# corpus back within reach from every language.
QUERY_TRANSLATION_ENABLED: bool = (
    os.environ.get("QUERY_TRANSLATION_ENABLED", "true").lower() != "false"
)
TRANSLATION_TIMEOUT: float = 6.0
TRANSLATION_CACHE_TTL: float = 600.0

# No single document may occupy more than this many candidate slots. The Italian
# catalogue holds 2155 chunks against ~520 datasheet chunks, so without a cap it
# took 16 of the 20 slots for any Italian question and no product datasheet was
# ever considered.
MAX_CHUNKS_PER_SOURCE = 2
# Distinct pages of one document that may contribute, so the overall ceiling per
# document is MAX_CHUNKS_PER_SOURCE * MAX_PAGES_PER_DOCUMENT.
MAX_PAGES_PER_DOCUMENT = 3

# A message this short is treated as a follow-up and searched together with the
# previous user turn, since on its own it names nothing to retrieve.
FOLLOWUP_MAX_CHARS = 60

# ---------------------------------------------------------------------------
# Relevance floor
# ---------------------------------------------------------------------------
# Minimum rerank score (0-10) for a chunk to enter the context. Without it the
# context is padded to FINAL_TOP_K even when only two chunks are on topic, and
# the filler measurably hurts: a question about the ITALICA 353 was answered
# correctly from its two datasheet chunks alone, but returned "I have no
# information" once four unrelated chunks — a privacy policy among them — were
# padded in alongside them.
RERANK_MIN_RELEVANCE = 4.0
# Never return fewer than this, so a harsh reranker cannot empty the context.
MIN_SOURCES = 2

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


# Ways a user asks for the same page in another language, in each of the four.
# "E in inglese?" is a request to change the link, not the prose, and the model
# kept returning the Italian URL because the context put that one first.
_URL_LANGUAGE_REQUEST: dict[str, tuple[str, ...]] = {
    "en": ("inglese", "english", "anglais", "inglés", "ingles"),
    "it": ("italiano", "italian", "italien"),
    "fr": ("francese", "french", "français", "francais", "francés", "frances"),
    "es": ("spagnolo", "spanish", "espagnol", "español", "espanol"),
}


def requested_url_language(message: str) -> Optional[str]:
    """Return the language the user explicitly asked the link to be in, if any."""
    lowered = _strip_accents(message.lower())
    for lang, terms in _URL_LANGUAGE_REQUEST.items():
        if any(_strip_accents(term) in lowered for term in terms):
            return lang
    return None


def _language_alternates(meta: dict) -> dict[str, str]:
    """
    Every language edition of the page this chunk came from.

    Only one was ever shown, so "and in English?" had no English URL to give and
    the model built one from the Italian slug. All four are already in the chunk
    metadata; nothing extra is fetched.
    """
    alternates = {}
    for lang in ("it", "en", "fr", "es"):
        url = meta.get(f"url_{lang}")
        if url and not _is_blocked_url(url):
            alternates[lang] = str(url)
    return alternates


def _pick_language_url(meta: dict, detected_lang: str) -> Optional[str]:
    """
    Return the page URL in the user's language from a chunk's metadata.

    Falls back through English then Italian then the canonical URL, so a page
    that exists in only some languages still produces a working link.
    """
    for key in (f"url_{detected_lang}", "url_en", "url_it", "canonical_url"):
        candidate = meta.get(key)
        if candidate:
            return str(candidate)
    return None


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
# Rerank cache: key -> (scores for the whole batch, timestamp)
# ---------------------------------------------------------------------------
_rerank_cache: dict[str, tuple[tuple[float, ...], float]] = {}


async def _score_batch(query: str, excerpts: list[str]) -> list[float]:
    """
    Score every candidate excerpt for *query* in ONE GPT call (0-10 each).

    Returns a list aligned with *excerpts*; entries are -1.0 where no usable
    score came back, and the caller falls back to the Pinecone score for those.
    A single call replaces the previous one-call-per-chunk fan-out, which was
    ~12x the cost and routinely hit the per-call timeout.
    """
    if not excerpts:
        return []

    cache_key = f"{hash(query)}:{hash(tuple(excerpts))}"
    now = time.monotonic()
    cached_entry = _rerank_cache.get(cache_key)
    if cached_entry is not None:
        scores, ts = cached_entry
        if now - ts < RERANK_CACHE_TTL:
            return list(scores)

    numbered = "\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(excerpts)
    )
    prompt = (
        "You rank retrieved documentation excerpts for a technical valve chatbot.\n"
        f"QUESTION: {query}\n\n"
        f"EXCERPTS:\n{numbered}\n\n"
        "Rate how useful each excerpt is for answering the QUESTION, from 0 "
        "(irrelevant) to 10 (directly answers it). Reward excerpts holding "
        "concrete technical data (sizes, pressures, materials, standards) that "
        "the question asks for.\n"
        "CSA publishes XLC sizes and weights once per range, not per model: an "
        "excerpt headed 'XLC 400 - …' documents every XLC 4xx model and one "
        "headed 'XLC 300 - …' every XLC 3xx model. So a dimensions table headed "
        "XLC 400 fully answers a question about an XLC 330/430 or XLC 310/410, "
        "and deserves a high score — do not mark it irrelevant because the exact "
        "model number is absent from it.\n"
        'Reply with JSON only: {"scores": {"0": <n>, "1": <n>, ...}} '
        "with one entry per excerpt index."
    )

    try:
        oai = _get_async_oai()
        resp = await asyncio.wait_for(
            oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            ),
            timeout=RERANK_TIMEOUT,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw).get("scores", {})
    except asyncio.TimeoutError:
        logger.warning("rerank: batched GPT scoring timed out (query='%s...')", query[:40])
        return [-1.0] * len(excerpts)
    except Exception as exc:
        logger.warning("rerank: batched GPT scoring failed: %s", exc)
        return [-1.0] * len(excerpts)

    scores: list[float] = []
    for i in range(len(excerpts)):
        value = parsed.get(str(i), parsed.get(i))
        try:
            scores.append(max(0.0, min(10.0, float(value))))
        except (TypeError, ValueError):
            scores.append(-1.0)

    _rerank_cache[cache_key] = (tuple(scores), now)  # type: ignore[assignment]
    return scores


async def rerank_chunks(query: str, chunks: list[tuple[str, float]]) -> list[tuple[float, int]]:
    """
    Rerank *chunks* with a single batched GPT-4o-mini call.

    Parameters
    ----------
    query  : user query string
    chunks : list of (text, pinecone_score) tuples

    Returns
    -------
    List of (final_score, original_index) sorted descending by final_score.
    """
    gpt_scores = await _score_batch(query, [text for text, _ in chunks])

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
# Function words shared with at least one sibling language. Worth 1 point: on
# their own they tie as often as they decide. "Parle-moi de la vanne … et
# donne-moi le lien" scored 2-2 between Italian and French on shared words
# alone, and the tie-break by dictionary order answered a French question in
# Italian, with an Italian link.
_LANG_HINTS: dict[str, list[str]] = {
    "it": ["il", "la", "le", "che", "per", "con", "del", "un", "una", "come", "questo"],
    "fr": ["le", "la", "les", "du", "des", "une", "pour", "avec", "est", "que"],
    "es": ["el", "la", "los", "las", "del", "para", "con", "una", "como", "que"],
    "en": ["the", "is", "are", "how", "what", "does", "can", "for", "with", "this", "that"],
}

# Words that belong to one of the four languages and not the others. Worth 3
# points, so a single unambiguous marker outweighs any number of shared ones.
_LANG_MARKERS: dict[str, list[str]] = {
    "it": ["quali", "quale", "dammi", "dove", "perché", "quanto", "quanti", "della", "degli",
           "sulla", "nel", "nella", "sono", "avete", "posso", "vostro", "vostra", "valvola",
           "valvole", "prezzo", "scheda", "sito", "grazie", "vorrei", "mi", "ci", "gli",
           # Content words: on a short technical question they are the only signal.
           "chi", "siamo", "azienda", "norme", "prodotti", "peso", "pressione", "portata",
           "sfiato", "sfiati", "materiali", "certificazioni", "consigli", "serve", "meglio",
           "vendete", "idranti", "fognatura", "acqua", "misure", "taglie", "link",
           "certificata", "certificato", "secondo", "norma", "garanzia", "anni", "offre",
           "valori", "diametri", "flangiata", "esercizio", "temperatura"],
    "fr": ["moi", "vers", "vos", "votre", "vous", "nous", "je", "quelles", "quels", "quelle",
           "quel", "donne", "parle", "dans", "sur", "pourquoi", "combien", "vanne", "vannes",
           "merci", "je voudrais", "c'est", "qu'est", "aussi", "cette", "ces", "sont",
           "materiaux", "poids", "revetement", "diametres", "pression", "debit", "ventouse",
           "entreprise", "societe", "gamme", "lien"],
    "es": ["cuáles", "cuál", "qué", "dónde", "cómo", "usted", "dame", "hacia", "sus", "tiene",
           "tienen", "válvula", "válvulas", "gracias", "quisiera", "sobre", "los", "esta",
           "estas", "hola", "puedo",
           "presion", "trabajo", "enlace", "maxima", "producto", "peso", "caudal", "ventosa",
           "empresa", "materiales", "cuerpo", "tapa", "obturador", "tienen ustedes"],
    "en": ["give", "me", "about", "tell", "which", "your", "please", "thanks", "valve",
           "valves", "size", "sizes", "material", "materials", "pressure", "do", "you",
           "weight", "flow", "link", "page", "company", "standards", "range", "datasheet"],
}

# Orthography that only one of the four uses.
_LANG_CHARS: dict[str, str] = {
    "es": "ñ¿¡",
    "fr": "çêîôû",
}

_WORD_SPLIT = re.compile(r"[^a-zàáâäçèéêëìíîïñòóôöùúûü']+")

# Language returned when the text carries no usable signal — a bare "e basta?",
# or a product code on its own. Answering such a turn in English because no
# Italian function word happened to appear switched the conversation's language
# mid-way; callers pass it to inherit the language instead.
UNKNOWN_LANGUAGE = "unknown"


def _strip_accents(text: str) -> str:
    """Fold diacritics so 'cuál' and 'cual' score the same."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(ch)
    )


def score_languages(text: str) -> dict[str, float]:
    """Return the per-language score for *text*; exposed for tests and tracing."""
    lower = text.lower()
    folded = _strip_accents(lower)
    words = {w for w in _WORD_SPLIT.split(folded) if w}

    scores: dict[str, float] = {}
    for lang in _LANG_HINTS:
        shared = sum(1 for w in _LANG_HINTS[lang] if _strip_accents(w) in words)
        exclusive = sum(3 for w in _LANG_MARKERS[lang] if _strip_accents(w) in words)
        # Orthography is checked on the original text: the diacritics are the signal.
        chars = 4 if any(c in lower for c in _LANG_CHARS.get(lang, "")) else 0
        scores[lang] = shared + exclusive + chars
    return scores


def detect_language(text: str, default: str = "en") -> str:
    """
    Heuristic language detection for it/en/fr/es.

    Scores shared function words at 1 point and language-exclusive markers at 3,
    then adds a bonus for orthography unique to one language.

    Returns *default* when nothing scores, and on a tie as well: resolving ties
    by dictionary order silently handed every ambiguous message to Italian.
    Pass default=UNKNOWN_LANGUAGE to detect that case instead of guessing.
    """
    scores = score_languages(text)
    best = max(scores.values())
    if best <= 0:
        return default

    winners = [lang for lang, score in scores.items() if score == best]
    if len(winners) > 1:
        return default
    return winners[0]


# ---------------------------------------------------------------------------
# Candidate weighting
# ---------------------------------------------------------------------------
def _adjusted_score(match: dict, detected_lang: str) -> float:
    """
    Weight a Pinecone match by chunk language and document priority.

    The same content exists in it/en/fr/es. Cosine similarity alone cannot tell
    the user's language apart, so a question asked in Italian often ranks the
    French translation of the right page above the Italian original.
    """
    score: float = match.get("score", 0.0)
    meta: dict = match.get("metadata", {})

    chunk_lang = (meta.get("lang") or "").lower()
    # With no idea what language the user wrote in, weighting by language would
    # be guessing: leave the cosine score alone rather than tilt it towards one.
    if chunk_lang and detected_lang != UNKNOWN_LANGUAGE:
        if chunk_lang == detected_lang:
            score *= LANG_MATCH_BOOST
        elif chunk_lang == "en":
            score *= LANG_ENGLISH_BOOST
        else:
            score *= LANG_MISMATCH_PENALTY
    # No 'lang' metadata: legacy English datasheets in docs/ — left neutral.

    if match.get("application_match"):
        score *= APPLICATION_MATCH_BOOST

    if match.get("exact_model_match"):
        score *= EXACT_MODEL_BOOST
    elif match.get("model_match"):
        score *= MODEL_MATCH_BOOST
        # Priority only breaks ties *within* the product the user named. Applied
        # unconditionally it made the XLC engineering set outrank every other
        # document for any query in its language: a question about where to find
        # the documentation filled all twelve reranker slots with XLC chunks and
        # never surfaced the documentation page at all.
        if meta.get("doc_priority"):
            score *= DOC_PRIORITY_BOOST

    return score


_translation_cache: dict[str, tuple[str, float]] = {}


async def translate_to_english(query: str) -> Optional[str]:
    """
    Return an English rendering of *query*, or None if unavailable.

    Used only to widen retrieval, never shown to the user, so a failure here
    degrades search quality but never the answer.
    """
    now = time.monotonic()
    cached = _translation_cache.get(query)
    if cached is not None and now - cached[1] < TRANSLATION_CACHE_TTL:
        return cached[0]

    try:
        oai = _get_async_oai()
        resp = await asyncio.wait_for(
            oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Translate this industrial-valve question into English. "
                            "Keep model codes and numbers unchanged. Reply with the "
                            f"translation only.\n\n{query}"
                        ),
                    }
                ],
                max_tokens=120,
                temperature=0,
            ),
            timeout=TRANSLATION_TIMEOUT,
        )
        english = (resp.choices[0].message.content or "").strip()
    except asyncio.TimeoutError:
        logger.warning("translation: timed out for query='%s...'", query[:40])
        return None
    except Exception as exc:
        logger.warning("translation: failed (%s)", exc)
        return None

    if not english:
        return None
    _translation_cache[query] = (english, now)
    return english


# Chunk ids end in "_c<index>" for every ingest script, so a chunk's neighbours
# on the same page are addressable by id without another similarity search.
_CHUNK_ID_TAIL = re.compile(r"^(?P<stem>.+_c)(?P<index>\d+)$")


def _neighbour_ids(chunk_id: str, span: int = 1) -> list[str]:
    """Ids of the chunks immediately before and after *chunk_id* on its page."""
    match = _CHUNK_ID_TAIL.match(chunk_id or "")
    if not match:
        return []
    stem, index = match.group("stem"), int(match.group("index"))
    return [
        f"{stem}{i}"
        for i in range(max(0, index - span), index + span + 1)
        if i != index
    ]


def _attach_neighbours(index, sources: list[Source]) -> None:
    """
    Append each source's adjacent chunks to its own text, in place.

    Chunking cuts mid-clause, and the sentence that answers the question is
    often in the piece next door: asked how long CSA's warranty runs, the model
    received the clause that says a defect must be reported within "8 giorni"
    but not the earlier line stating the term, and reported an eight-year
    warranty. The neighbours are completions of a source already chosen, not new
    candidates — they never enter ranking, never occupy a context slot, and are
    fetched by id rather than by similarity.
    """
    wanted: dict[str, list[str]] = {}
    for src in sources:
        ids = _neighbour_ids(src.chunk_id, NEIGHBOUR_SPAN)
        if ids:
            wanted[src.chunk_id] = ids

    all_ids = sorted({i for ids in wanted.values() for i in ids})
    if not all_ids:
        return

    fetched: dict[str, str] = {}
    try:
        # Pinecone caps the request URI length, so ids go up in small batches.
        for start in range(0, len(all_ids), 40):
            response = index.fetch(ids=all_ids[start : start + 40])
            for vid, vector in (response.vectors or {}).items():
                text = (vector.metadata or {}).get("text", "")
                if text:
                    fetched[vid] = text
    except Exception as exc:
        logger.warning("neighbours: fetch failed (%s)", exc)
        return

    for src in sources:
        extra = [fetched[i] for i in wanted.get(src.chunk_id, []) if i in fetched]
        if not extra:
            continue
        combined = "\n".join([src.text_full, *extra])
        src.text_full = combined[:MAX_CONTEXT_CHARS_WITH_NEIGHBOURS]


# A series designation as it appears in a question and in a chunk heading:
# "XLC 300", "ITALICA 353", "FOX 3F".
_SERIES = re.compile(r"\b([A-Za-z]{2,})\s*(\d{2,4})\b")

# Prefixes that look like a series but are sizes and ratings: "DN 100" names a
# diameter and "PN 16" a pressure class, on any product whatsoever. Counting
# them as series marked the FOX family's catalogue table as THE datasheet of an
# ITALICA question — both carried "DN 100" — and the bot answered the FOX's
# 26 kg with the ITALICA table two sources below.
_NON_SERIES_PREFIXES = {"dn", "pn"}


def _series_designations(text: str) -> set[str]:
    """Series named in *text*, sizes and ratings excluded: {'xlc 300'}."""
    return {
        f"{m.group(1).lower()} {m.group(2)}"
        for m in _SERIES.finditer(text)
        if m.group(1).lower() not in _NON_SERIES_PREFIXES
    }


# "DN 300", "DN300", "diametro 300" — the size a table-lookup question is about.
_DN_IN_QUERY = re.compile(r"\bdn\s*(\d{2,4})\b", re.I)


# Every way the four languages ask about weight, as a noun *and as a verb*.
# "Quanto pesa la XLC 400 DN 300?" is the commonest phrasing there is, and
# matching only the noun ("peso", "pesi") left it recognised by nothing below:
# the question got no requested label, so the pinning that exists precisely to
# rescue a weights table never ran and the answer came back "non ho il dato".
_WEIGHT_WORDS = (
    r"\bpes[aoi]\b|\bpes(?:ano|an|os)\b"      # it/es: peso, pesi, pesa, pesano, pesan
    r"|\bweigh(?:t|ts|s|ed)?\b"               # en: weight, weighs, weigh
    r"|\bpoids\b|\bp[eè]se(?:nt)?\b"          # fr: poids, pèse, pèsent
)

# Asking for weights or overall dimensions is asking for the dimensions table,
# whether or not a particular size is named. "Quota" (singular) matters on its
# own: "Qual è la quota A della XLC 330 DN 250?" matched nothing here while
# "le quote" did, and the dimensions table lost the pool to the version pages.
_TABLE_SUBJECT = re.compile(
    rf"{_WEIGHT_WORDS}|\bmisur[ae]\b|\bdimension|\bquot[ae]\b|\btagli[ae]\b|\bingombr",
    re.I,
)


# The column label a table uses for each thing a question can ask about. Used to
# pin a chunk that demonstrably holds the figure, instead of trusting the
# reranker to recognise a table: asked for the XLC 330/430's weights it scored
# the pages headed "Dati tecnici" above the pages that actually carry the weight
# column, and the answer came back "not documented".
#
# The column patterns must cover every spelling ingest/pdf_extract.py actually
# produces — these are tested against the chunk text, and the corpus is not
# uniform: the XLC engineering tables write "Weight (Kg)" / "Peso (Kg)" with
# parentheses, the product datasheets write "Weight Kg" without, the ATHENA
# sheet abbreviates to "Wt Kg", and the VRCD sheet drops the space
# ("Weight(Kg)"). An exact-string list silently un-pins whichever sheets it
# doesn't spell: with only "Weight Kg" listed, no XLC engineering page could
# ever be pinned, and with only "Weight (Kg)", no datasheet could.
_ATTRIBUTE_LABELS: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    (re.compile(_WEIGHT_WORDS, re.I),
     # Anche il peso scritto in prosa: la GEMINA FF non ha tabella e dichiara
     # "Weight 2,3 Kg." in mezzo al testo, quindi il pin per colonna non lo
     # vedeva e il peso restava senza risposta con la riga in contesto.
     re.compile(r"(?:Weight|Wt|Peso|Poids)\s?(?:\(?Kg\)?|[\d.,]+\s*Kg)")),
    (re.compile(r"\bkv\b", re.I), re.compile(r"Kv \(m3/h\)")),
    (re.compile(r"\bcorsa\b|\bstroke\b", re.I),
     re.compile(r"(?:Corsa|Stroke|Course|Carrera)[^=;\n]{0,14}\(mm\)")),
    (re.compile(r"\bportat[ae]\b|\bflow rate\b", re.I),
     # La colonna porta spesso un qualificatore prima dell'unita' — la tabella
     # della ATHENA scrive "Flow rate max. (l/s)" — e pretendere l'unita'
     # attaccata al nome escludeva proprio le tabelle di portata: la riga della
     # ATHENA 1"-1 1/4" non entrava in contesto e il modello leggeva un numero
     # dal testo speculare della barra laterale.
     re.compile(r"(?:Portata|Flow rate|Débit|Caudal)[^=;\n]{0,14}\(l/s\)")),
    # A quota question names a lettered column of the dimensions table. Weight
    # questions were pinned to the table and quota questions were not, so
    # "quota A della XLC 330 DN 250" refused while "quanto pesa" answered —
    # same table, same page.
    (re.compile(r"\bquot[ae]\b|\bdimensioni?\b|\bmisur[ae]\b|\bdimensions?\b", re.I),
     re.compile(r"\b[ABCDE] ?\(mm\) ?=")),
    # Pressure and temperature live in the "Working conditions" prose of the
    # model's own sheet, not in a table: without a pin those pages lost the
    # pool to other products' pages and 12 documented values were refused.
    (re.compile(
        r"\bpression[ei]\b|\bpressure\b|\bpresi[oó]n\b|\bpression\b", re.I),
     re.compile(
         r"(?:Max(?:imum)?|Min(?:imum)?)\.?\s+(?:operating\s+|static\s+)?pressure", re.I)),
    (re.compile(r"temperatur", re.I),
     re.compile(r"(?:Maximum |Max\.? )?temperature[: ]|max\. \d+\s?°C", re.I)),
)


def _requested_labels(query: str) -> tuple[re.Pattern, ...]:
    """Column patterns for the table columns the question asks for, if any."""
    return tuple(
        column for pattern, column in _ATTRIBUTE_LABELS if pattern.search(query)
    )


def _series_asked(query: str) -> set[str]:
    """Series designations the query names, lowercased: {'xlc 300'}."""
    return _series_designations(query)


def _heading_names_series(text: str, asked: set[str]) -> bool:
    """True when the chunk's own first line names a series in *asked*."""
    heading = (text or "").split("\n", 1)[0]
    return bool(_series_designations(heading) & asked)


def _order_holders(items: list, match_of, query: str) -> list:
    """
    Stable-sort column-holding candidates by who has the right to answer.

    The named model's own datasheet first, then its family, then chunks whose
    heading names a series the query asks about, then pool order. Pool order
    alone hands the pinned slots to whoever happens to sit first: asked what
    the FOX SUB weighs, the restore step had prepended two catalogue tables
    that also carry a weight column, the pin took those, and the answer came
    from another product's table while FOX_SUB.pdf's own — present in the pool,
    five positions down — was never pinned.
    """
    # For pin ordering the asked series includes the XLC *range* of any model
    # named: "XLC 330" reads its dimensions from the table headed "XLC 300",
    # and matching designations literally pinned the 400-series table instead —
    # the model then, correctly, refused to read it. Range expansion stays out
    # of _mark_series_match: marking the range pages as THE datasheet is
    # exactly what reported the series' 25 bar for a 16 bar variant.
    asked = _series_asked(query) | {
        f"xlc {m.group(1)}00" for m in _XLC_MODEL.finditer(query)
    }

    def text_of(item) -> str:
        return match_of(item).get("metadata", {}).get("text", "")

    def page_series(item) -> set[str]:
        meta = match_of(item).get("metadata", {})
        return series_on_page(meta.get("source_file", ""), meta.get("page"))

    def rank(item) -> tuple:
        m = match_of(item)
        on_page = page_series(item) if asked else set()
        # A page that documents another series answers a different valve, and
        # the chunk itself often cannot say so: the XLC 600 weight table is
        # titled "[Table p.5]" and names nothing, so it lost the pinned slots
        # to a catalogue page of the XLC 500 and the DN 100's weight went
        # unanswered. The page map settles it either way.
        return (
            bool(on_page) and not (asked & on_page),
            not m.get("exact_model_match"),
            not m.get("model_match"),
            not (asked and (asked & on_page or _heading_names_series(text_of(item), asked))),
        )

    if all(rank(item) == (False, True, True, True) for item in items):
        return items
    return sorted(items, key=rank)


def _pin_chunks_holding(
    candidates: list[dict], labels: tuple[re.Pattern, ...], limit: int, query: str = ""
) -> list[int]:
    """
    Indices of the best candidates whose text actually contains *labels*.

    Among the holders, precedence follows _order_holders: the named model's own
    chunks, then its family's, then the ones whose heading names the series
    asked about. Both the XLC 300 and the XLC 400 dimension tables carry a
    "Peso (Kg)" column, and pinning by pool order alone pinned the 400-series
    table for a question about an XLC 300 — its cosine score is a shade
    higher — so the model, correctly refusing to read a 400 table for a 300,
    answered that it had no figure while page 20 carried it.
    """
    if not labels:
        return []
    found: list[int] = []
    for index, candidate in enumerate(candidates):
        text = candidate.get("metadata", {}).get("text", "")
        if any(label.search(text) for label in labels):
            found.append(index)
    found = _order_holders(found, lambda i: candidates[i], query)
    return found[:limit]


def _query_all_vectors(
    index, vectors: list, top_k: int, filter_: dict, namespace: Optional[str] = None
) -> list:
    """
    Run one filtered search per search vector and keep each chunk's best score.

    The searches differ by more than translation: one of the vectors renders the
    question in the shape the serialised tables use, and that is the only one a
    row of figures matches well. Searching with a single vector left the page
    holding the weights out of reach.
    """
    extra = {"namespace": namespace} if namespace else {}
    best: dict[str, object] = {}
    try:
        for vector in vectors:
            found = index.query(
                vector=vector, top_k=top_k, include_metadata=True, filter=filter_, **extra
            )
            for m in found.get("matches", []):
                mid = m.get("id", "")
                previous = best.get(mid)
                if previous is None or m.get("score", 0.0) > previous.get("score", 0.0):
                    best[mid] = m
    except Exception as exc:
        logger.warning("filtered query failed (%s)", exc)
    return list(best.values())


def _table_row_probe(query: str) -> Optional[str]:
    """
    Render a size question the way the serialised tables are written.

    ingest/pdf_extract.py emits one self-describing line per size — "DN (mm) =
    300; A (mm) = 850; … Peso (Kg) = 405" — and a prose question embeds poorly
    against it. Searching additionally with that shape brings the row itself
    into reach.
    """
    sizes = _DN_IN_QUERY.findall(query)
    if sizes:
        heads = " ".join(f"DN (mm) = {size}" for size in dict.fromkeys(sizes))
        return f"{heads}; A (mm); B (mm); Kv (m3/h); Peso (Kg)"
    if _TABLE_SUBJECT.search(query):
        return "DN (mm); A (mm); B (mm); C (mm); D (mm); E (mm); Peso (Kg)"
    return None


# An XLC model number and the range it belongs to: XLC 430 -> XLC 400.
_XLC_MODEL = re.compile(r"\bxlc\s*([3456])(\d{2})\b", re.I)


def _note_range_coverage(sources: list[Source], query: str) -> None:
    """
    State, on the source itself, that a range table covers the model asked about.

    CSA publishes XLC dimensions once per range, so the table answering a
    question about an XLC 330/430 is headed "XLC 300" or "XLC 400" and never
    names the model. Told only as a general rule, the model would not apply it:
    the rules against mixing series — the ones that stop a variant's pressure
    being reported for the base valve — legitimately point the other way. Saying
    it as a fact about this particular source resolves the conflict without
    weakening those rules.
    """
    # The claim holds for sizes and weights ONLY — that is what CSA publishes
    # per range. Working pressures are per model: the range's standard version
    # takes 25 bar while the XLC 353, 380/480 and 310 ND stop at 16, and with
    # this note active on a pressure question the bot answered 25 bar for all
    # three — a component rated 16 bar reported at 25. So the note fires only
    # when the question asks for table figures (weights, dimensions, a DN size).
    if not (_TABLE_SUBJECT.search(query) or _DN_IN_QUERY.search(query)):
        return
    ranges = {f"{m.group(1)}00" for m in _XLC_MODEL.finditer(query)}
    if not ranges:
        return

    models_by_range: dict[str, list[str]] = {}
    for m in _XLC_MODEL.finditer(query):
        models_by_range.setdefault(f"{m.group(1)}00", []).append(
            f"XLC {m.group(1)}{m.group(2)}"
        )

    for src in sources:
        heading = (src.text_full or "").split("\n", 1)[0]
        on_page = series_on_page(src.source_file, src.page)
        for range_number, model_names in models_by_range.items():
            if (
                re.search(rf"\bXLC\s*{range_number}\b", heading, re.I)
                # Il chunk-tabella della XLC 600 si intitola "[Table p.5]"
                # e non nomina la serie: senza la mappa restava senza nota
                # e il peso del DN 100 non veniva risposto.
                or f"xlc {range_number}" in on_page
            ):
                names = " and ".join(dict.fromkeys(model_names))
                src.context_note = (
                    f"RANGE TABLE FOR THE MODEL ASKED ABOUT — CSA publishes XLC sizes "
                    f"and weights once per range, and this XLC {range_number} table is "
                    f"the one that covers {names}. Its figures ARE {names}'s figures; "
                    f"answer from it."
                )
                break


def _mark_foreign_sections(sources: list[Source], query: str) -> None:
    """
    Demote the pages of a multi-product file that document a different product.

    APOLLO_RPC.pdf documents the Apollo RP on pages 8-9 and the Apollo RPC on
    10-11; SCS_AS.pdf appends the GOLIA SUB kit at page 5. The exact-model
    banner is per file, so it endorsed both sections alike and the generator
    took whichever table looked richer: the Apollo RPC's quota A came back as
    the RP's 682 mm, and the SCS-AS's weight as the SUB kit's 7,0-88,3 kg,
    with the right rows sitting in the context unmarked.

    Only files the section map knows are touched, and only when the query names
    one of their sections — otherwise there is nothing to disambiguate.
    """
    named = find_sections(query)
    if not named:
        return

    for src in sources:
        wanted = named.get(src.source_file)
        if not wanted:
            continue
        actual = section_of(src.source_file, src.page)
        if actual is None:
            continue
        if actual == wanted:
            # Say it affirmatively too. The RPC's own table labels its rows
            # "RP 80X ... RP 80D" — the RPC name appears nowhere inside the
            # table — so with only the generic banner the model would not
            # attribute the rows to the RPC and refused a quota it was
            # looking at.
            if not src.context_note:
                src.context_note = (
                    f"This page of {src.source_file} documents the {wanted}, the "
                    f"model asked about. Every figure on it is the {wanted}'s, "
                    "whatever short form the table's own row labels use."
                )
            continue
        src.is_exact_model = False
        src.context_note = (
            f"DIFFERENT PRODUCT — this page of {src.source_file} documents the "
            f"{actual}, not the {wanted} the question is about. The file covers "
            "both. Never take figures from here for the model asked about."
        )


def _mark_series_match(sources: list[Source], query: str) -> None:
    """
    Flag the sources whose own heading names the series the question asks about.

    One document can hold several ranges: the XLC engineering manual documents
    the XLC 400 on one set of pages and the XLC 300 on another, and each chunk
    opens with the heading saying which. Asked for an XLC 300's weight the model
    read the XLC 400's table even with the right chunk first in the context, so
    the distinction is stated rather than left to be inferred from the heading.
    """
    asked = _series_designations(query)
    if not asked:
        return

    for src in sources:
        heading = (src.text_full or "").split("\n", 1)[0]
        if _series_designations(heading) & asked:
            src.is_exact_model = True


def _cap_per_source(matches: list[dict], limit: int = MAX_CHUNKS_PER_SOURCE) -> list[dict]:
    """
    Limit how many candidate slots any single document, and any single page of
    it, may take.

    *matches* must be sorted best-first. Capping per document alone treats a
    21-page engineering manual like a 3-page datasheet: every page of the XLC
    engineering document competed for the same four slots, and the page holding
    the dimensions table lost to the pages holding the Kv tables, so the weight
    of a DN 300 was reported as undocumented while it sat on page 12.
    """
    kept: list[dict] = []
    per_document: dict[str, int] = {}
    per_page: dict[tuple[str, object], int] = {}

    for m in matches:
        meta = m.get("metadata", {})
        source = meta.get("source_file", "")
        # Crawled pages all report 'csasrl.it'; count them per page instead so
        # the cap does not collapse the entire website into four chunks.
        if source == "csasrl.it":
            source = meta.get("canonical_url", source)

        page_key = (source, meta.get("page"))
        if per_page.get(page_key, 0) >= limit:
            continue
        if per_document.get(source, 0) >= limit * MAX_PAGES_PER_DOCUMENT:
            continue

        per_document[source] = per_document.get(source, 0) + 1
        per_page[page_key] = per_page.get(page_key, 0) + 1
        kept.append(m)
    return kept


def _translation_key(match: dict) -> Optional[str]:
    """
    Identity of a chunk across languages, or None when it has no translations.

    Web pages share a canonical URL across languages; the XLC engineering PDFs
    share a doc_group plus page number.
    """
    meta: dict = match.get("metadata", {})
    canonical = meta.get("canonical_url")
    if canonical:
        return f"url:{canonical}:{meta.get('chunk_index', '')}"
    doc_group = meta.get("doc_group")
    if doc_group:
        return f"doc:{doc_group}:p{meta.get('page', '')}:{meta.get('chunk_index', '')}"
    return None


def _select_with_reserved_slots(
    ranked: list[tuple[float, int]],
    candidates: list[dict],
    requested_labels: tuple[str, ...] = (),
    query: str = "",
) -> list[int]:
    """
    Pick FINAL_TOP_K candidate indices, reserving slots for model matches.

    The reranker judges relevance from a 600-char excerpt, so it can rate a
    chatty catalogue passage above the terse table that actually holds the
    figure. When the user named a model, up to MODEL_RESERVED_SLOTS chunks from
    that model's datasheet are kept regardless of where the reranker put them.
    """
    ordered = [idx for _, idx in ranked]
    relevant = [idx for score, idx in ranked if score >= RERANK_MIN_RELEVANCE]

    # Keep a floor of sources even when the reranker scores everything low.
    if len(relevant) < MIN_SOURCES:
        relevant = ordered[:MIN_SOURCES]

    exact_indices = [i for i in ordered if candidates[i].get("exact_model_match")]
    model_indices = [i for i in ordered if candidates[i].get("model_match")]
    application_indices = [i for i in ordered if candidates[i].get("application_match")]

    if not model_indices and not application_indices:
        return relevant[:FINAL_TOP_K]

    limit = APPLICATION_FINAL_TOP_K if application_indices else FINAL_TOP_K

    # A chunk that literally carries the column asked for answers the question,
    # whatever the reranker made of it.
    selected = _pin_chunks_holding(candidates, requested_labels, ATTRIBUTE_PINNED_SLOTS, query)

    # Then the datasheet the user actually named, then its siblings.
    for idx in exact_indices[:EXACT_MODEL_RESERVED_SLOTS]:
        if idx not in selected:
            selected.append(idx)
    for idx in model_indices[:MODEL_RESERVED_SLOTS]:
        if idx not in selected:
            selected.append(idx)

    # Reserve slots for application matches from *different* products: several
    # chunks of one datasheet would answer "which valves suit X" with one valve.
    seen_sources: set[str] = set()
    for idx in application_indices:
        if len(seen_sources) >= APPLICATION_RESERVED_SLOTS:
            break
        meta = candidates[idx].get("metadata", {})
        source = meta.get("source_file", "")
        # Le pagine del sito si chiamano tutte 'csasrl.it': contate per nome
        # file valevano come un prodotto solo, e i due primi idranti si
        # prendevano tutti gli slot lasciando fuori il terzo. Ogni pagina
        # prodotto e' un prodotto, come lo e' ogni scheda.
        if source == "csasrl.it":
            source = meta.get("canonical_url", source)
        if source in seen_sources or idx in selected:
            continue
        seen_sources.add(source)
        selected.append(idx)

    for idx in relevant:
        if len(selected) >= limit:
            break
        if idx not in selected:
            selected.append(idx)

    # Restore reranker order among the chosen chunks so the strongest leads.
    return sorted(selected[:limit], key=ordered.index)


def _dedupe_translations(matches: list[dict]) -> list[dict]:
    """
    Keep only the best-scoring language variant of each translated chunk.

    *matches* must already be sorted by descending adjusted score, so the first
    occurrence of a translation key is the one to keep.
    """
    kept: list[dict] = []
    seen_keys: set[str] = set()
    for m in matches:
        key = _translation_key(m)
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        kept.append(m)
    return kept


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def build_search_query(message: str, history: Optional[list] = None) -> str:
    """
    Expand a context-free follow-up using the conversation it belongs to.

    Retrieval sees only the current message, so a follow-up like "e basta?" or
    "e in acciaio inox?" searches for words that name no product and comes back
    with nothing — the model then reports it has no information, moments after
    answering the same topic. Short follow-ups are therefore searched together
    with the previous user turn.

    A short follow-up that names a product is NOT expanded: it already carries
    its own subject, and gluing the previous turn on destroys it. The registry
    keeps only the most specific model named, so "athena che valvola è?" asked
    after a question about the XLC 300 searched as "…XLC 300 DN 300 athena che
    valvola è?" — where "xlc 300" (two tokens) beats "athena" (one) — and the
    bot answered a question about the ATHENA from XLC documents, then "lynx?"
    from ATHENA ones, refusing both moments after answering fluently.
    """
    if not history or len(message) > FOLLOWUP_MAX_CHARS:
        return message

    # The message names a product on its own: it is a topic change, not a
    # continuation, and it searches best exactly as written.
    if find_exact_model_source(message) or find_model_sources(message):
        return message

    previous = [
        h.content for h in history
        if getattr(h, "role", None) == "user" and h.content
    ]
    if not previous:
        return message
    return f"{previous[-1]} {message}"


def resolve_language(message: str, history: Optional[list]) -> str:
    """
    Language to answer in: the message's own, or the conversation's when the
    message carries no signal.

    A turn like "e basta?" or "e in acciaio inox?" scores zero in every language,
    and defaulting it to English answered an Italian conversation in English —
    with an English product link to match. Walking back through the user's own
    turns keeps the thread in one language, while any message with a real signal
    still switches it, so a user who changes language is followed.
    """
    own = detect_language(message, default=UNKNOWN_LANGUAGE)
    if own != UNKNOWN_LANGUAGE:
        return own

    for turn in reversed(history or []):
        if getattr(turn, "role", None) != "user" or not turn.content:
            continue
        earlier = detect_language(turn.content, default=UNKNOWN_LANGUAGE)
        if earlier != UNKNOWN_LANGUAGE:
            return earlier

    return UNKNOWN_LANGUAGE


def _is_followup(message: str, history: Optional[list]) -> bool:
    """True when *message* is a short turn leaning on the conversation."""
    return bool(history) and len(message) <= FOLLOWUP_MAX_CHARS


def _products_already_covered(history: Optional[list]) -> set[str]:
    """Datasheets named in the assistant's most recent answer."""
    for turn in reversed(history or []):
        if getattr(turn, "role", None) == "assistant" and turn.content:
            return sources_mentioned(turn.content)
    return set()


async def retrieve(
    query: str,
    top_k: int = TOP_K,
    language_hint: Optional[str] = None,
    history: Optional[list] = None,
) -> tuple[list[Source], str]:
    """
    Embed *query*, query Pinecone, optionally rerank with GPT-4o-mini,
    and return (sources, detected_language).

    Parameters
    ----------
    query          : user message text
    top_k          : number of chunks to return (ignored when RERANK_ENABLED;
                     FINAL_TOP_K is used instead)
    language_hint  : lingua della pagina che ospita il widget; usata solo
                     quando il messaggio non ha segnale proprio

    Returns
    -------
    sources            : list[Source] sorted by score descending
    detected_language  : 'it' | 'en' | 'fr' | 'es'
    """
    # Language is detected from what the user actually typed, falling back to
    # the conversation when the turn is too short to judge. The search itself
    # runs on the follow-up expanded with its conversation context.
    # Il suggerimento (la lingua della pagina che ospita il widget) vale solo
    # dove il messaggio non dice nulla: "Cos'e' l'APOLLO RPC SMART?" e' quasi
    # tutto nome di prodotto, il rilevatore restituiva 'unknown' e il link
    # ricadeva sull'inglese in mezzo a un sito italiano. Imporlo invece
    # spezzerebbe la regola opposta, gia' pagata: chi scrive in un'altra lingua
    # va seguito, anche su una pagina italiana.
    detected_lang = resolve_language(query, history)
    if detected_lang == UNKNOWN_LANGUAGE and language_hint:
        detected_lang = language_hint
    # The prose follows the conversation's language; the link follows whatever
    # language the user asked the page to be in, when they asked at all.
    url_language = requested_url_language(query) or detected_lang
    search_query = build_search_query(query, history)
    oai, index = _get_clients()

    # Search with the question as asked and, for non-English questions, with an
    # English rendering too — the product datasheets are English-only.
    search_texts = [search_query]
    if QUERY_TRANSLATION_ENABLED and detected_lang != "en":
        english = await translate_to_english(search_query)
        if english and english.lower() != search_query.lower():
            search_texts.append(english)

    # A question about one size is answered by a table row, and a sentence of
    # prose embeds poorly against a row of figures: asked what an XLC 400 DN 300
    # weighs, the page holding the weights never entered the candidate pool. So
    # the size is also searched for in the form the serialised tables use.
    table_probe = _table_row_probe(search_query)
    if table_probe:
        search_texts.append(table_probe)

    # Embed every search text in one call (sync SDK — fast, no benefit from async)
    emb_response = oai.embeddings.create(model=EMBED_MODEL, input=search_texts)
    query_vectors = [item.embedding for item in emb_response.data]
    query_vector = query_vectors[0]

    # How many to fetch from Pinecone
    fetch_k = RERANK_TOP_K if RERANK_ENABLED else top_k
    _fetch_k = fetch_k * 3  # over-fetch to allow merge/dedup across namespaces

    raw_matches: list = []
    for vector in query_vectors:
        raw_matches += list(
            index.query(vector=vector, top_k=_fetch_k, include_metadata=True).get("matches", [])
        )
        raw_matches += list(
            index.query(
                vector=vector, top_k=_fetch_k, include_metadata=True, namespace="catalog"
            ).get("matches", [])
        )

    # Third query: when the question names a specific valve model, fetch that
    # model's datasheet by filename. Semantic search cannot separate "353" from
    # "310", so without this the right document is often never a candidate.
    # The datasheet whose own model code the question names, fetched separately
    # from its family. Sharing MODEL_MATCH_TOP_K with eight sibling variants let
    # the reranker drop the base model's own datasheet — "maximum pressure" reads
    # as closer to a document about high pressure — and the FOX 3F was answered
    # at the FOX 3F-HP's 64 bar instead of its own 40.
    exact_file = find_exact_model_source(search_query)
    raw_exact_matches: list = []
    if exact_file:
        raw_exact_matches = _query_all_vectors(
            index, query_vectors, EXACT_MODEL_TOP_K, {"source_file": {"$eq": exact_file}}
        )
        logger.debug("exact model: %s -> %d chunks", exact_file, len(raw_exact_matches))

    # Range-level documents covering the model named. The individual XLC
    # datasheets carry no dimensions — sizes and weights are published once per
    # range — so without these a question about an XLC 330/430's weight found
    # only that model's own datasheet and came back "not documented".
    model_files = find_model_sources(search_query)
    series_files = [f for f in find_series_documents(search_query) if f not in model_files]
    if series_files:
        model_files = model_files + series_files

    # Products documented inside another product's file are invisible to the
    # registry, which is built from file names: the CSFL flow regulator is a
    # section of XLC_PILOTS.pdf, so its dimensions were answered "I have no
    # information" while the file was never even fetched.
    section_files = [f for f in files_with_sections(search_query) if f not in model_files]
    if section_files:
        model_files = model_files + section_files

    # The range documents get their own slots. Sharing MODEL_MATCH_TOP_K with
    # the whole family let them lose every one: asked what an XLC 600 DN 100
    # weighs, the eight slots went to XLC 3xx datasheets and XLC_500_SIZING.pdf
    # — the only document with a 600 table — never reached the pool at all.
    raw_series_matches: list = []
    if series_files:
        raw_series_matches = _query_all_vectors(
            index, query_vectors, MODEL_MATCH_TOP_K,
            {"source_file": {"$in": series_files}},
        )
        model_files = [f for f in model_files if f not in series_files]

    raw_model_matches: list = []
    if model_files:
        raw_model_matches = _query_all_vectors(
            index, query_vectors, MODEL_MATCH_TOP_K, {"source_file": {"$in": model_files}}
        )
        logger.debug(
            "model match: query names %s -> %d chunks",
            model_files, len(raw_model_matches),
        )

    # Some families publish their dimensions only in the general catalogue:
    # not one ITALICA datasheet carries a dimension table, while the catalogue's
    # "Italica 300 - Dimensioni e pesi" page does — and nothing else links that
    # page to the model name, so "dimensioni della ITALICA 353" was refused with
    # the table sitting in the index. On a table-figure question that names a
    # family, the family's own catalogue pages are fetched as candidates too.
    # Gated on the table probe so ordinary questions widen nothing.
    named_family = find_family(search_query) if table_probe else None
    if named_family:
        raw_family_catalogue = _query_all_vectors(
            index, query_vectors, MODEL_MATCH_TOP_K,
            {"product_family": {"$eq": named_family.upper()}},
            namespace="catalog",
        )
        if raw_family_catalogue:
            logger.debug(
                "family catalogue: %s -> %d chunks",
                named_family, len(raw_family_catalogue),
            )
            raw_model_matches = raw_model_matches + raw_family_catalogue

    raw_model_matches = raw_series_matches + raw_model_matches

    def _mark_family_catalogue_pages(sources: list[Source]) -> None:
        """
        Say, on the source itself, that a catalogue page belongs to the family
        asked about — and is therefore authoritative, not a rival product.

        The ITALICA case needs both halves. Its dimensions exist only on the
        catalogue's family pages, and the generic "DIFFERENT PRODUCT" label made
        the bot refuse figures it was looking at; but exempting the catalogue
        wholesale swung the failure the other way — the reranker put the FOX
        family's catalogue weight table first for an ITALICA question, and with
        no label to stop it the bot answered the FOX's 26 kg.

        A family, though, is not a series. The catalogue devotes separate pages
        to the XLC 300, 500 and 600, all of them "XLC" pages: affirming them
        alike told the model that the XLC 500's table answered a question about
        an XLC 330, and its DN 80 came back as 20 kg instead of 24.
        """
        if not named_family:
            return

        # The series the question is about, XLC model numbers expanded to their
        # range: an XLC 330 reads the pages headed "XLC 300".
        asked = _series_designations(query) | {
            f"xlc {m.group(1)}00" for m in _XLC_MODEL.finditer(query)
        }
        asked = {d for d in asked if d.split()[0] == named_family}

        for src in sources:
            if not is_catalogue(src.source_file):
                continue
            if (src.product_family or "").lower() != named_family:
                continue

            # Which series the catalogue page documents, from the page map:
            # the chunk itself is often a wall of figures naming no series, and
            # catalogue chunk ids do not end in "_c<N>" so no neighbour is ever
            # attached to supply it.
            on_page = {
                d for d in series_on_page(src.source_file, src.page)
                if d.split()[0] == named_family
            }
            if asked and on_page and not (asked & on_page):
                src.context_note = (
                    f"CATALOGUE PAGE OF A DIFFERENT SERIES — this page documents "
                    f"the {'/'.join(sorted(d.upper() for d in on_page))}, not the "
                    f"{'/'.join(sorted(d.upper() for d in asked))} the question is "
                    "about. Never take its figures for the model asked about."
                )
                continue

            src.context_note = (
                f"FAMILY CATALOGUE PAGE FOR THE MODEL ASKED ABOUT — the "
                f"{named_family.upper()} family publishes these figures once "
                "for the whole series, and they apply to the model asked "
                "about; answer from them."
            )

    # Fourth query: a question about an application ("what do you recommend for
    # irrigation?") is answered by a *set* of products. Similarity alone returns
    # several chunks of the one or two closest datasheets, so the datasheets
    # documented for that application are pulled in as candidates too — 32 name
    # irrigation, and the bot was listing two of them.
    # Le caratteristiche costruttive si comportano come le applicazioni: la
    # risposta e' un insieme di prodotti, non uno solo.
    feature_files = find_feature_sources(search_query)

    application_files = find_application_sources(" ".join(search_texts))

    # On a follow-up, skip the products the previous answer already covered.
    # Without this "e basta?" retrieves the same datasheets as the question it
    # follows, the model finds nothing new to say and reports that it has no
    # information — while 34 datasheets document irrigation.
    if feature_files:
        application_files = sorted(set(application_files) | set(feature_files))

    if application_files and _is_followup(query, history):
        already = _products_already_covered(history)
        remaining = [f for f in application_files if f not in already]
        if remaining:
            application_files = remaining

    raw_application_matches: list = []
    if application_files:
        raw_application_matches = _query_all_vectors(
            index, query_vectors, APPLICATION_MATCH_TOP_K,
            {"source_file": {"$in": application_files}},
        )
        logger.debug(
            "application match: %d datasheets -> %d chunks",
            len(application_files), len(raw_application_matches),
        )

    # Pagine prodotto del sito della categoria nominata. L'elenco che il bot
    # produce a "quali idranti fate?" nasce dall'indice delle schede: un
    # prodotto che il sito pubblica ma nessuna scheda indicizzata documenta
    # non compariva affatto — l'APOLLO RPC SMART rispondeva quando lo si
    # nominava, ma spariva dagli elenchi. Poche pagine, una per prodotto, e
    # solo quando la domanda nomina davvero una categoria.
    category_urls = find_category_products(search_query)
    raw_category_matches: list = []
    if category_urls:
        raw_category_matches = _query_all_vectors(
            index, query_vectors, APPLICATION_MATCH_TOP_K,
            {"canonical_url": {"$in": category_urls[:60]}},
        )
        logger.debug(
            "category: %d pagine prodotto -> %d chunk",
            len(category_urls), len(raw_category_matches),
        )

    exact_ids = {m.get("id", "") for m in raw_exact_matches}
    model_ids = {m.get("id", "") for m in raw_model_matches}
    application_ids = {m.get("id", "") for m in raw_application_matches}
    application_ids |= {m.get("id", "") for m in raw_category_matches}

    # Copy each match into a plain dict: the Pinecone SDK returns ScoredVector
    # objects that reject new keys, and the weighting below annotates them.
    # A chunk found by several searches keeps its best score.
    seen_ids: set[str] = set()
    matches: list[dict] = []
    for m in (raw_exact_matches + raw_model_matches + raw_category_matches
              + raw_application_matches + raw_matches):
        mid = m.get("id", "")
        score = m.get("score", 0.0)
        if mid in seen_ids:
            for existing in matches:
                if existing["id"] == mid:
                    existing["score"] = max(existing["score"], score)
                    break
            continue
        seen_ids.add(mid)
        matches.append(
            {
                "id": mid,
                "score": score,
                "metadata": dict(m.get("metadata") or {}),
                "model_match": mid in model_ids or mid in exact_ids,
                "exact_model_match": mid in exact_ids,
                "application_match": mid in application_ids,
            }
        )

    # Filter by MIN_SCORE, but never drop a chunk from the datasheet the user
    # named: it is the authoritative answer even at a modest cosine score.
    # Chunks reached through a filtered query are exempt from the score floor:
    # they come from datasheets the registry documents for exactly this model or
    # application, so their relevance is established by the registry rather than
    # by cosine similarity. A product for desalination sat at 0.294 against the
    # 0.30 floor and was discarded before the reserved slots could keep it.
    filtered: list[dict] = [
        m for m in matches
        if m.get("score", 0.0) >= MIN_SCORE
        or m.get("model_match")
        or m.get("application_match")
    ]

    # Apply language / document-priority weighting, then re-sort. This runs
    # before the GPT reranker so the reranker sees the best candidates.
    for m in filtered:
        m["adjusted_score"] = _adjusted_score(m, detected_lang)
    filtered.sort(key=lambda m: m["adjusted_score"], reverse=True)

    # Drop same-page duplicates that differ only by language, keeping the
    # best-scoring variant — otherwise one page can occupy every context slot.
    filtered = _dedupe_translations(filtered)

    # Stop any one document from monopolising the candidate pool, tightening the
    # limit on application questions where the answer should span products.
    requested_labels = _requested_labels(search_query)
    capped = _cap_per_source(
        filtered,
        # Una domanda di categoria vuole ampiezza fra prodotti quanto una
        # domanda per applicazione: col tetto largo la scheda dell'Apollo RPC
        # si prendeva quattro slot e la pagina del terzo idrante non entrava.
        APPLICATION_MAX_CHUNKS_PER_SOURCE
        if (application_files or category_urls)
        else MAX_CHUNKS_PER_SOURCE,
    )

    # Put back any chunk that carries the column the question asks for. Those
    # pages lose the cap to better-phrased neighbours — the XLC dimensions pages
    # rank below the pages headed "Dati tecnici" — and then the only chunks that
    # could answer a question about weights are gone before anything is ranked.
    if requested_labels:
        already = {id(m) for m in capped}
        holders = [
            m for m in filtered
            if id(m) not in already
            and any(
                label.search(m.get("metadata", {}).get("text", ""))
                for label in requested_labels
            )
        ]
        # Same precedence as the pin: restoring another product's table for a
        # question that names a model puts the wrong figures one slot ahead of
        # the right ones.
        holders = _order_holders(holders, lambda m: m, search_query)
        restored = holders[:ATTRIBUTE_PINNED_SLOTS]
        # Prepended, not appended: the pool is trimmed to RERANK_TOP_K right
        # after this, so anything added at the end is dropped again.
        capped = restored + capped
    filtered = capped

    # Cap to RERANK_TOP_K (or top_k when reranking disabled) before scoring
    pre_rerank = filtered[:fetch_k]

    # -----------------------------------------------------------------------
    # Optional GPT reranking
    # -----------------------------------------------------------------------
    if RERANK_ENABLED and pre_rerank:
        try:
            chunk_texts = [
                (
                    m.get("metadata", {}).get("text", "")[:RERANK_EXCERPT_CHARS],
                    m.get("adjusted_score", m.get("score", 0.0)),
                )
                for m in pre_rerank
            ]
            ranked = await rerank_chunks(search_query, chunk_texts)
            top_indices = _select_with_reserved_slots(
                ranked, pre_rerank, requested_labels, search_query
            )
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

        # Pick the URL in the user's language — or in the language they asked the
        # link to be in, when they said so ("e in inglese?").
        url = _pick_language_url(meta, url_language)

        # Discard URLs that match blocked patterns (shop/cart/checkout/account pages)
        if _is_blocked_url(url):
            url = None

        full_text = meta.get("text", "")
        sources.append(
            Source(
                source_file=meta.get("source_file", "unknown"),
                page=meta.get("page"),
                chunk_id=meta.get("chunk_id", match["id"]),
                score=round(score, 4),
                text_snippet=full_text[:200],
                text_full=full_text[:MAX_CONTEXT_CHARS_PER_SOURCE],
                url=url or None,
                product_family=meta.get("product_family") or None,
                valve_model=meta.get("valve_model") or None,
                page_title=meta.get("page_title") or None,
                lang=(meta.get("lang") or None),
                applications=applications_for(meta.get("source_file", "")),
                features=features_for(meta.get("source_file", "")),
                is_exact_model=bool(match.get("exact_model_match")),
                url_alternates=_language_alternates(meta),
            )
        )

    _mark_series_match(sources, search_query)
    _note_range_coverage(sources, search_query)
    _mark_foreign_sections(sources, search_query)

    # Le pagine prodotto della categoria sono spesso l'unica fonte di un
    # prodotto: il modello costruiva l'elenco dalle schede tecniche e si
    # fermava li'. In contesto la pagina c'era in 3 esecuzioni su 3, ma
    # l'APOLLO RPC SMART compariva nella risposta solo in una. Come per le
    # applicazioni documentate, la cosa si risolve dicendola come un fatto su
    # questa fonte, non come regola generale.
    if category_urls:
        insieme = set(category_urls)
        for src in sources:
            if src.context_note:
                continue
            url = src.url or ""
            alternative = set(src.url_alternates.values()) | {url}
            if insieme & alternative:
                src.context_note = (
                    "PRODOTTO DELLA CATEGORIA CHIESTA — questa pagina del sito "
                    "documenta un prodotto che appartiene alla categoria della "
                    "domanda. Se stai elencando i prodotti di quella categoria, "
                    "includilo: puo' essere l'unica fonte che lo nomina."
                )

    # The datasheet of the model actually named leads the context. Both the
    # FOX 3F and the FOX 3F-C hold a "Flanged 200" row, and when the variant's
    # row came first the model answered the FOX 3F's weight with the variant's
    # 92,0 kg instead of its own 85,0 kg — the right source was present but read
    # second.
    # Within the exact matches, the datasheet leads the page from csasrl.it:
    # both can be about exactly the model asked, but the site page is marketing
    # copy — for the XLC 353 it lists "Pressione: 10-16-25 bar" (the available
    # PN classes) while the datasheet states the operating limit, 16 bar. With
    # the site page first, the bot reported a 16 bar valve at 25.
    sources.sort(
        key=lambda s: (
            not s.is_exact_model,
            not s.source_file.lower().endswith(".pdf"),
        )
    )

    if NEIGHBOUR_SPAN:
        _attach_neighbours(index, sources)

    # After the neighbours: a catalogue table chunk is a wall of figures that
    # names no series, while the prose chunk beside it says "valvole XLC 500".
    # Marking before the merge, that evidence was invisible and the XLC 500
    # page was affirmed as the XLC 330's own.
    _mark_family_catalogue_pages(sources)

    # When reranking is disabled, enforce score-descending order
    if not RERANK_ENABLED:
        sources.sort(key=lambda s: s.score, reverse=True)

    # Ultimo, cosi' nessun riordino li sposta: i programmi di calcolo aprono il
    # contesto quando la domanda li riguarda.
    _prepend_sizing_programs(sources, search_query, detected_lang)

    return sources, detected_lang


# ---------------------------------------------------------------------------
# Programmi di calcolo
# ---------------------------------------------------------------------------
# Le pagine dei programmi stanno dietro il login, quindi in Pinecone c'e' solo
# un puntatore che non dice quali valvole ciascuno dimensioni — e con quel
# puntatore fuori dal contesto la domanda "c'e' un calcolatore per la valvola
# AUGUSTA?" riceveva un rifiuto seguito dal link al calcolatore XLC: il
# programma di un'altra famiglia, senza dire che era di un'altra famiglia.
# Dimensionare una valvola col programma sbagliato e' un problema di sicurezza.
#
# Le fonti qui sotto non vengono dall'indice: sono costruite da
# api/sizing_programs.json, che e' verificato pagina per pagina. Cosi' la
# risposta non dipende da quando e' stata rifatta la scansione del sito.
MAX_SIZING_SOURCES = 2

_ACCESSO_TESTO = {
    "login": {
        "it": "richiede la registrazione e l'accesso all'area clienti di csasrl.it",
        "en": "requires registration and login to the csasrl.it customer area",
        "fr": "nécessite l'inscription et la connexion à l'espace client de csasrl.it",
        "es": "requiere registro e inicio de sesión en el área de clientes de csasrl.it",
    },
    "pubblica": {
        "it": "si usa liberamente, senza registrazione",
        "en": "is freely usable, no registration needed",
        "fr": "s'utilise librement, sans inscription",
        "es": "se usa libremente, sin registro",
    },
}


def _testo_programma(programma: dict, lang: str) -> tuple[str, dict[str, str]]:
    """Le righe di contesto di un programma, e i suoi URL per lingua."""
    per_lingua = {p["lingua"]: p for p in programma["pagine"]}
    alternates = {ling: p["url"] for ling, p in per_lingua.items()}
    scelta = per_lingua.get(lang) or per_lingua.get("it") or next(iter(per_lingua.values()))

    def tradotto(campo: str) -> str:
        valori = programma.get(campo, {})
        return valori.get(lang) or valori.get("it") or next(iter(valori.values()), "")

    righe = [
        f"Programma di calcolo CSA: {tradotto('nome')}",
        f"Dimensiona: {tradotto('copre')}",
    ]
    if programma.get("famiglie"):
        righe.append("Serie coperte: " + ", ".join(programma["famiglie"]))
    accesso = _ACCESSO_TESTO.get(scelta["accesso"], {})
    righe.append("Accesso: " + (accesso.get(lang) or accesso.get("it", "")))
    righe.append(f"Link: {scelta['url']}")
    # Il calcolatore XLC e' pubblico in inglese, francese e spagnolo ma protetto
    # in italiano: dirlo evita di mandare un italiano su un form di login
    # quando la stessa cosa gli e' accessibile in un'altra lingua.
    if scelta["accesso"] == "login":
        pubbliche = [p for p in programma["pagine"] if p["accesso"] == "pubblica"]
        if pubbliche:
            righe.append(
                "Nota: l'edizione in questa lingua richiede il login, mentre "
                "queste sono pubbliche: "
                + ", ".join(f"{p['lingua']} {p['url']}" for p in pubbliche)
            )
    return "\n".join(righe), alternates


def _prepend_sizing_programs(sources: list[Source], query: str, lang: str) -> None:
    """Mette in testa al contesto i programmi di calcolo pertinenti."""
    trovati = find_sizing_programs(query)
    if not trovati:
        return

    nuove: list[Source] = []
    for programma in trovati["pertinenti"][:MAX_SIZING_SOURCES]:
        testo, alternates = _testo_programma(programma, lang)
        nuove.append(Source(
            source_file="web_scraper",
            page=None,
            chunk_id=f"sizing_{programma['chiave']}",
            score=1.0,
            text_snippet=testo[:200],
            text_full=testo,
            url=alternates.get(lang) or alternates.get("it") or next(iter(alternates.values())),
            url_alternates=alternates,
            page_title=programma["nome"].get(lang) or programma["nome"].get("it"),
            lang=lang,
            context_note=(
                "PROGRAMMA DI CALCOLO CHE COPRE LA VALVOLA CHIESTA — rispondi "
                "che questo programma esiste e dallo con il suo link. Nomina "
                "sempre, nella stessa frase, le serie che dimensiona."
            ),
        ))

    # L'elenco completo va sempre, anche quando un programma pertinente c'e':
    # e' cio' che permette di rispondere "per questa valvola non risulta un
    # programma, esistono questi" invece del rifiuto secco.
    righe = [
        "Elenco completo dei programmi di calcolo pubblicati da CSA "
        "(nessun altro programma esiste sul sito):"
    ]
    alternates_elenco: dict[str, str] = {}
    for programma in trovati["tutti"]:
        testo, alternates = _testo_programma(programma, lang)
        righe.append("- " + testo.replace("\n", " | "))
        for ling, url in alternates.items():
            alternates_elenco.setdefault(f"{programma['chiave']}_{ling}", url)
    for voce in trovati.get("indice_sezione", []):
        if voce["lingua"] == lang:
            righe.append(f"Pagina indice della sezione dimensionamento: {voce['url']}")
            alternates_elenco.setdefault("indice", voce["url"])

    nuove.append(Source(
        source_file="web_scraper",
        page=None,
        chunk_id="sizing_elenco",
        score=1.0,
        text_snippet=righe[0][:200],
        text_full="\n".join(righe),
        url=None,
        url_alternates=alternates_elenco,
        page_title="Programmi di calcolo CSA",
        lang=lang,
        context_note=(
            "ELENCO COMPLETO E VERIFICATO dei programmi di calcolo CSA. Se la "
            "valvola della domanda non compare fra le serie coperte da nessuna "
            "riga, dillo apertamente e nomina i programmi che esistono: NON "
            "offrire il programma di un'altra serie come se andasse bene per "
            "lei. Dimensionare una valvola col programma sbagliato e' un "
            "problema di sicurezza."
        ),
    ))

    sources[:0] = nuove


# ---------------------------------------------------------------------------
# Context builder — converts sources to a formatted string for the prompt
# ---------------------------------------------------------------------------
# Le caratteristiche sono registrate con un nome inglese, perche' il corpus lo
# e'. Dichiararlo cosi' nel contesto bastava a una domanda inglese e non a una
# italiana: "avete valvole convogliate?" continuava a ricevere un rifiuto con
# le quattro schede giuste davanti, perche' nulla legava "convogliate" a
# "conveyed air discharge". La riga porta quindi anche i termini italiani, e
# la lettera di modello con cui il cliente le chiama.
FEATURE_LABELS: dict[str, str] = {
    "conveyed air discharge":
        "conveyed air discharge — in italiano: scarico dell'aria convogliato, "
        "valvola convogliata (sono i modelli con suffisso C e i kit SUB)",
    "anti-shock":
        "anti-shock — in italiano: anti-colpo d'ariete, non-slam (modelli AS)",
    "anti-surge":
        "anti-surge — in italiano: anti-colpo d'ariete, antiariete (modelli RFP)",
    "remote monitoring":
        "remote monitoring — in italiano: controllo remoto, telecontrollo "
        "(modelli SMART)",
}


def build_context_string(sources: list[Source], detected_lang: str) -> str:
    """
    Format retrieved sources into a context block for the system prompt.

    Uses each source's *full* chunk text. Feeding the 200-char UI snippet here
    silently discarded ~90% of every retrieved chunk, which made the model
    answer "I have no information" on questions the corpus did answer.
    """
    if not sources:
        return ""

    # When the datasheet of the exact model is present, every other datasheet in
    # the context is a variant of it and is labelled as such. Labelling only the
    # right one was not enough: asked for the FOX 3F's DN 100 weight, the model
    # saw 21,7 kg once and the variants' 26 kg twice, and followed the majority.
    has_exact = any(s.is_exact_model for s in sources)

    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        lines.append(f"--- Source {i} (score={src.score}) ---")
        if src.is_exact_model and src.source_file.lower().endswith(".pdf"):
            lines.append(
                "THIS IS THE DATASHEET OF EXACTLY THE MODEL ASKED ABOUT — take the "
                "figures from here, not from a variant of it."
            )
        elif src.is_exact_model:
            # The model's own page on csasrl.it: right subject, lower authority.
            # It lists commercial ranges ("Pressione: 10-16-25 bar" = the PN
            # classes on offer) where the datasheet states operating limits, and
            # with the datasheet banner on it the bot reported a 16 bar valve at
            # 25. Descriptions and links yes, figures only if no datasheet says
            # otherwise.
            lines.append(
                "OFFICIAL PRODUCT PAGE of the model asked about, from csasrl.it. "
                "For technical figures the datasheet excerpts take precedence "
                "over this page: when they disagree, answer with the datasheet's "
                "figure. Use this page for descriptions, applications and links."
            )
        elif has_exact and is_catalogue(src.source_file):
            # A catalogue page marked as the asked-about family's own carries
            # its affirmative note below. Any other catalogue page is another
            # family's — the reranker put the FOX weight table first for an
            # ITALICA question, and without this label its 26 kg was answered.
            if not src.context_note:
                lines.append(
                    "CATALOGUE PAGE OF A DIFFERENT PRODUCT FAMILY"
                    + (f" ({src.product_family})" if src.product_family else "")
                    + " — NOT the model asked about; never take its figures for "
                    "the model asked about."
                )
        elif (
            has_exact
            and src.source_file.lower().endswith(".pdf")
            and not src.context_note
        ):
            # A source carrying an affirmative note (range table for the model
            # asked about) is never simultaneously forbidden: printing both
            # "answer from it" and "never take its figures" on the same source
            # left the model refusing quota questions the table answered.
            lines.append(
                f"DIFFERENT PRODUCT — this is the datasheet of {src.source_file}, a "
                "variant, NOT the model asked about. Use it only if the user asks "
                "about this variant; never take its figures for the model asked about."
            )
        if src.source_file != "web_scraper":
            lines.append(f"File: {src.source_file}, Page: {src.page}")
        if src.page_title:
            lines.append(f"Page title: {src.page_title}")
        if src.context_note:
            lines.append(src.context_note)
        if src.applications:
            lines.append(
                f"Applications documented in this datasheet: {', '.join(src.applications)}"
            )
        if src.features:
            lines.append(
                "Construction features documented in this datasheet: "
                + "; ".join(FEATURE_LABELS.get(f, f) for f in src.features)
            )
        if src.url:
            lines.append(f"URL ({detected_lang}): {src.url}")
        others = {
            lang: url for lang, url in src.url_alternates.items() if url != src.url
        }
        # L'elenco dei programmi di calcolo porta gli URL di otto pagine diverse,
        # non le traduzioni di una: annunciarli come "la stessa pagina in altre
        # lingue" sarebbe falso, e ripeterli raddoppierebbe il contesto — sono
        # gia' tutti nel testo, riga per riga accanto al programma che aprono.
        # Restano in url_alternates perche' e' da li' che si costruisce la lista
        # dei link che la risposta puo' citare.
        if others and src.chunk_id != "sizing_elenco":
            lines.append(
                "Same page in other languages: "
                + ", ".join(f"{lang}={url}" for lang, url in sorted(others.items()))
            )
        lines.append(src.text_full or src.text_snippet)
        lines.append("")

    return "\n".join(lines)
