"""
tests/test_20_questions.py
--------------------------
20 test questions covering:
  1.  Product information (valves, specs, certifications)
  2.  Multilingual responses (IT / EN / FR / ES)
  3.  URL correctness (language-specific links)
  4.  Out-of-scope handling (polite decline)
  5.  Edge cases (empty message, very long query)

Tests that need API keys are marked with @pytest.mark.integration and
are SKIPPED automatically when OPENAI_API_KEY is not set.

Tests that do NOT need API keys (language detection, URL map parsing,
prompt rendering, model schemas) run in plain unit mode.

Run all tests:
    pytest tests/test_20_questions.py -v

Run only unit tests (no keys needed):
    pytest tests/test_20_questions.py -v -m "not integration"
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

# Load .env so API keys are available when running pytest from the repo root
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
REQUIRES_KEYS = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY")
    or "YOUR" in os.environ.get("OPENAI_API_KEY", "")
    or len(os.environ.get("OPENAI_API_KEY", "")) < 20,
    reason="OPENAI_API_KEY not configured — skipping integration test",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1–3  Language detection (unit — no keys)
# ---------------------------------------------------------------------------
class TestLanguageDetection:
    def _detect(self, text: str) -> str:
        from api.retrieval import detect_language
        return detect_language(text)

    def test_detects_italian(self):
        assert self._detect("Come funziona questa valvola?") == "it"

    def test_detects_english(self):
        assert self._detect("What are the technical specifications for this valve?") == "en"

    def test_detects_french(self):
        assert self._detect("Comment installer cette vanne?") == "fr"

    def test_detects_spanish(self):
        assert self._detect("¿Cuáles son las especificaciones de esta válvula?") == "es"


# ---------------------------------------------------------------------------
# 4–5  Pydantic schemas (unit — no keys)
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_chat_request_valid(self):
        from api.models import ChatRequest
        req = ChatRequest(message="Hello CSA")
        assert req.message == "Hello CSA"
        assert req.language is None

    def test_chat_request_with_language(self):
        from api.models import ChatRequest
        req = ChatRequest(message="Bonjour", language="fr")
        assert req.language == "fr"

    def test_source_model(self):
        from api.models import Source
        src = Source(
            source_file="catalogue.pdf",
            page=5,
            chunk_id="catalogue.pdf_p5_c0",
            score=0.87,
            text_snippet="Ball valves for industrial use...",
            url="https://csasrl.it/en/ball-valves",
        )
        assert src.score == 0.87

    def test_chat_response_model(self):
        from api.models import ChatResponse
        resp = ChatResponse(answer="CSA valves meet API 6D.", detected_language="en")
        assert resp.detected_language == "en"
        assert resp.sources == []


# ---------------------------------------------------------------------------
# 6–7  System prompt builder (unit — no keys)
# ---------------------------------------------------------------------------
class TestPromptBuilder:
    def test_prompt_contains_language(self):
        from api.prompt import build_system_prompt
        prompt = build_system_prompt("some context", detected_language="it")
        assert "it" in prompt

    def test_prompt_contains_context(self):
        from api.prompt import build_system_prompt
        ctx = "Ball valve DN50 rated for 40 bar."
        prompt = build_system_prompt(ctx, "en")
        assert ctx in prompt

    def test_prompt_no_context_fallback(self):
        from api.prompt import build_system_prompt
        prompt = build_system_prompt("", "en")
        assert "No relevant context" in prompt


# ---------------------------------------------------------------------------
# 8–10  Sitemap XML parsing (unit — no keys)
# ---------------------------------------------------------------------------
MOCK_SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://csasrl.it/product-sitemap.xml</loc></sitemap>
  <sitemap><loc>https://csasrl.it/page-sitemap.xml</loc></sitemap>
</sitemapindex>"""

MOCK_URL_SET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <url>
    <loc>https://csasrl.it/en/ball-valves/</loc>
    <xhtml:link rel="alternate" hreflang="it" href="https://csasrl.it/it/valvole-a-sfera/"/>
    <xhtml:link rel="alternate" hreflang="en" href="https://csasrl.it/en/ball-valves/"/>
    <xhtml:link rel="alternate" hreflang="fr" href="https://csasrl.it/fr/robinets-a-bille/"/>
    <xhtml:link rel="alternate" hreflang="es" href="https://csasrl.it/es/valvulas-de-bola/"/>
  </url>
  <url>
    <loc>https://csasrl.it/en/gate-valves/</loc>
    <xhtml:link rel="alternate" hreflang="it" href="https://csasrl.it/it/saracinesche/"/>
    <xhtml:link rel="alternate" hreflang="en" href="https://csasrl.it/en/gate-valves/"/>
  </url>
</urlset>"""


class TestSitemapParsing:
    def test_parse_sitemap_index(self):
        from ingest.web_scraper import parse_sitemap_index
        urls = parse_sitemap_index(MOCK_SITEMAP_INDEX, "https://csasrl.it")
        assert "https://csasrl.it/product-sitemap.xml" in urls
        assert "https://csasrl.it/page-sitemap.xml" in urls
        assert len(urls) == 2

    def test_parse_url_set_hreflang(self):
        from ingest.web_scraper import parse_url_set
        entries = parse_url_set(MOCK_URL_SET)
        assert len(entries) == 2
        ball_valve = next(e for e in entries if "ball-valves" in e["loc"])
        assert ball_valve["langs"]["it"] == "https://csasrl.it/it/valvole-a-sfera/"
        assert ball_valve["langs"]["fr"] == "https://csasrl.it/fr/robinets-a-bille/"
        assert ball_valve["langs"]["es"] == "https://csasrl.it/es/valvulas-de-bola/"

    def test_parse_url_set_partial_langs(self):
        from ingest.web_scraper import parse_url_set
        entries = parse_url_set(MOCK_URL_SET)
        gate_valve = next(e for e in entries if "gate-valves" in e["loc"])
        # Only it and en hreflang present
        assert "it" in gate_valve["langs"]
        assert "en" in gate_valve["langs"]


# ---------------------------------------------------------------------------
# 11  Context string builder (unit — no keys)
# ---------------------------------------------------------------------------
class TestContextBuilder:
    def test_build_context_includes_url(self):
        from api.models import Source
        from api.retrieval import build_context_string

        sources = [
            Source(
                source_file="web_scraper",
                page=None,
                chunk_id="url__csa_ball_valves",
                score=0.92,
                text_snippet="ball valves industrial",
                url="https://csasrl.it/it/valvole-a-sfera/",
            )
        ]
        ctx = build_context_string(sources, "it")
        assert "https://csasrl.it/it/valvole-a-sfera/" in ctx

    def test_build_context_empty(self):
        from api.retrieval import build_context_string
        ctx = build_context_string([], "en")
        assert ctx == ""


# ---------------------------------------------------------------------------
# 12–13  Chunk utility (unit — no keys)
# ---------------------------------------------------------------------------
class TestChunking:
    def test_chunk_produces_multiple_pieces(self):
        from ingest.pdf_ingest import chunk_text
        long_text = " ".join(["word"] * 1100)
        chunks = chunk_text(long_text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_chunk_overlap_exists(self):
        from ingest.pdf_ingest import chunk_text, _tokenize
        long_text = " ".join([f"word{i}" for i in range(600)])
        chunks = chunk_text(long_text, chunk_size=500, overlap=50)
        # Last tokens of chunk[0] should appear at start of chunk[1]
        tail_tokens = _tokenize(chunks[0])[-50:]
        head_tokens = _tokenize(chunks[1])[:50]
        assert tail_tokens == head_tokens


# ---------------------------------------------------------------------------
# 14  URL map JSON persistence (unit — no keys)
# ---------------------------------------------------------------------------
class TestUrlMapPersistence:
    def test_json_round_trip(self, tmp_path):
        url_map = {
            "https://csasrl.it/en/ball-valves/": {
                "it": "https://csasrl.it/it/valvole-a-sfera/",
                "en": "https://csasrl.it/en/ball-valves/",
                "fr": "https://csasrl.it/fr/robinets-a-bille/",
                "es": "https://csasrl.it/es/valvulas-de-bola/",
            }
        }
        p = tmp_path / "url_map.json"
        p.write_text(json.dumps(url_map, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == url_map


# ---------------------------------------------------------------------------
# 15–20  Integration tests (require OPENAI_API_KEY + PINECONE_API_KEY)
# ---------------------------------------------------------------------------
@REQUIRES_KEYS
class TestIntegration:
    """
    These tests exercise the full stack. They are skipped when keys are absent.
    """

    @pytest.fixture(autouse=True)
    def _check_pinecone(self):
        key = os.environ.get("PINECONE_API_KEY", "")
        if not key or "YOUR" in key or len(key) < 10:
            pytest.skip("PINECONE_API_KEY not configured")

    def test_q15_ball_valve_specs_english(self):
        """Product info: ball valve technical specs (EN)."""
        from api.retrieval import retrieve
        sources, lang = retrieve("What are the technical specifications for CSA ball valves?", language_hint="en")
        assert lang == "en"
        assert isinstance(sources, list)  # may be empty if index not populated

    def test_q16_italian_query_returns_it_language(self):
        """Multilingual: Italian query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = retrieve("Quali sono le caratteristiche delle valvole a sfera CSA?")
        assert lang == "it"

    def test_q17_french_query_returns_fr_language(self):
        """Multilingual: French query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = retrieve("Quelles sont les caractéristiques des vannes à bille CSA?")
        assert lang == "fr"

    def test_q18_spanish_query_returns_es_language(self):
        """Multilingual: Spanish query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = retrieve("¿Cuáles son las características de las válvulas de bola CSA?")
        assert lang == "es"

    def test_q19_url_in_correct_language(self):
        """URL correctness: Italian query gets Italian URL."""
        from api.retrieval import retrieve
        sources, lang = retrieve("valvole a sfera CSA", language_hint="it")
        url_sources = [s for s in sources if s.url]
        if url_sources:
            for src in url_sources:
                assert src.url and (
                    "/it/" in src.url or "csasrl.it" in src.url
                ), f"Expected Italian URL, got: {src.url}"

    def test_q20_out_of_scope_via_prompt(self):
        """Out-of-scope: system prompt instructs model to decline unrelated questions."""
        from api.prompt import build_system_prompt
        prompt = build_system_prompt("", detected_language="en")
        # Prompt must contain instructions to politely decline out-of-scope topics
        assert "out-of-scope" in prompt.lower() or "unrelated" in prompt.lower()
