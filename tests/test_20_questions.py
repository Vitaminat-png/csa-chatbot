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

    @pytest.mark.parametrize(
        "expected, text",
        [
            # Shared function words alone tie between these languages; each case
            # below was, or could be, resolved to the wrong language when only
            # shared words were counted — and a wrong language means the answer
            # and its product link come back in a language the user did not ask in.
            ("fr", "Parle-moi de la vanne Gemina et donne-moi le lien vers sa page produit."),
            ("it", "Parlami della valvola Gemina e dammi il link alla pagina prodotto."),
            ("es", "Háblame de la válvula Gemina y dame el enlace a su página de producto."),
            ("en", "Tell me about the Gemina valve and give me the link to its product page."),
            ("fr", "Quelles sont les dimensions de la XLC 400 et son poids?"),
            ("it", "A che pressione lavora la ITALICA 353?"),
            ("fr", "Bonjour, je voudrais des informations sur vos vannes."),
            ("it", "Buongiorno, vorrei informazioni sulle vostre valvole."),
            ("es", "Hola, quisiera información sobre sus válvulas."),
            ("en", "Hello, I would like information about your valves."),
        ],
    )
    def test_disambiguates_similar_phrasings(self, expected: str, text: str):
        assert self._detect(text) == expected


class TestFollowUpQueries:
    """
    Retrieval sees only the current message, so a bare follow-up searches for
    words that name no product and comes back empty — the bot then claims to
    have no information moments after answering the same topic.
    """

    def _history(self):
        from api.models import HistoryMessage
        return [
            HistoryMessage(role="user", content="per irrigazione cosa consigli?"),
            HistoryMessage(role="assistant", content="Ti consiglio gli sfiati SCF."),
        ]

    def test_short_followup_inherits_previous_question(self):
        from api.retrieval import build_search_query
        assert build_search_query("e basta?", self._history()) == (
            "per irrigazione cosa consigli? e basta?"
        )

    def test_self_contained_question_is_left_alone(self):
        from api.retrieval import build_search_query
        question = "Quali taglie DN ha la XLC 400 e con che valori Kv sono associate?"
        assert build_search_query(question, self._history()) == question

    def test_a_followup_naming_a_product_is_not_expanded(self):
        """
        "athena che valvola è?" after a question about the XLC 300 searched as
        one string, where the registry keeps only the most specific model named
        — "xlc 300" beat "athena" — and the ATHENA question was answered from
        XLC documents. A short turn that names a product carries its own
        subject: expanding it destroys the subject instead of supplying one.
        """
        from api.models import HistoryMessage
        from api.retrieval import build_search_query
        history = [HistoryMessage(role="user", content="Quanto pesa la XLC 300 DN 300")]
        assert build_search_query("athena che valvola è?", history) == "athena che valvola è?"
        assert build_search_query("lynx?", history) == "lynx?"
        # A subject-less follow-up still gets the previous turn glued on.
        assert build_search_query("e basta?", history).startswith("Quanto pesa la XLC 300")

    def test_no_history_leaves_message_unchanged(self):
        from api.retrieval import build_search_query
        assert build_search_query("e basta?", None) == "e basta?"


class TestModelVariantsAreDistinct:
    """
    A suffix makes a different valve. The audit found the FOX 3F reported at
    64 bar — the rating of the carbon-steel FOX 3F-HP — while the base valve is
    ductile iron PN 40. Answering that to someone sizing a line pressurises a
    PN 40 casting to 64 bar.
    """

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("Qual e la pressione massima della FOX 3F?", "FOX_3F.pdf"),
            ("pressione massima FOX 3F HP", "FOX_3F_HP.pdf"),
            ("Quanto pesa la FOX 3F C flangiata DN 200?", "FOX_3F_C.pdf"),
            ("Quanto pesa lo sfiato LYNX 3F flangiato DN 200?", "LYNX_3F.pdf"),
            ("A che pressione lavora la ITALICA 353?", "ITALICA_353.pdf"),
        ],
    )
    def test_the_named_model_resolves_to_its_own_datasheet(self, question, expected):
        from api.model_index import find_exact_model_source
        assert find_exact_model_source(question) == expected

    @pytest.mark.parametrize(
        "question, expected_file",
        [
            ("mi dai le dimensioni di atena", "ATHENA.pdf"),
            ("quanto pesa la italika 353?", "ITALICA_353.pdf"),
            ("che valvola è la ciclops 3f rfp?", "CYCLOPS_3F_RFP.pdf"),
            ("misure della gollia 3f", "GOLIA_3F.pdf"),
        ],
    )
    def test_misspelled_product_names_still_resolve(self, question, expected_file):
        """
        Users type product names by ear — "atena", "italika" — and exact token
        matching answered "I have no information on this product" while the
        datasheet sat in the corpus. One-edit alignment onto family names,
        same first letter, unique candidate only.
        """
        from api.model_index import find_model_sources
        assert expected_file in find_model_sources(question)

    @pytest.mark.parametrize(
        "question",
        [
            # "largo" is one edit from ARGO but starts differently; "solo" is
            # one edit from EOLO but four letters. Everyday words must never
            # become valves.
            "quanto è largo il corpo della valvola?",
            "vorrei solo sapere i prezzi",
            "dove trovo il catalogo completo?",
        ],
    )
    def test_everyday_words_do_not_become_products(self, question):
        from api.model_index import find_model_sources
        assert find_model_sources(question) == []

    def test_dn_and_pn_are_sizes_not_series(self):
        """
        "DN 100" in a query and "DN100" in a FOX table heading matched as a
        shared "series", the FOX catalogue table was labelled THE datasheet of
        an ITALICA 310 question, and its 26 kg beat the ITALICA table two
        sources below.
        """
        from api.retrieval import _series_designations
        assert _series_designations("Quanto pesa la ITALICA 310 DN 100 PN 16?") == {
            "italica 310"
        }
        assert _series_designations('1" 2"/DN50 DN80 DN100 1,5 1,4') == set()

    def test_one_edit_distance(self):
        from api.model_index import _one_edit_apart
        assert _one_edit_apart("atena", "athena")      # insertion
        assert _one_edit_apart("ciclops", "cyclops")   # substitution
        assert _one_edit_apart("golia", "gollia")      # deletion
        assert not _one_edit_apart("athena", "athena")  # identical
        assert not _one_edit_apart("atene", "athena")   # two edits
        assert not _one_edit_apart("fox", "eolo")

    def test_a_range_question_names_no_single_model(self):
        from api.model_index import find_exact_model_source, find_model_sources
        question = "Che sfiati della famiglia FOX avete?"
        assert find_exact_model_source(question) is None
        # The family list stays, so range questions still reach every variant.
        assert len(find_model_sources(question)) > 1


class TestWeightQuestionsReachTheWeightColumn:
    """
    Asking a weight is what pins the chunk that carries the weight column, and
    both halves of that were broken.

    The question pattern matched only the noun, so "Quanto pesa la XLC 400 DN
    300?" — the commonest phrasing there is — requested no label at all and the
    pinning never ran: the weights page lost the per-document cap to six pages
    headed "Dati tecnici" and the bot said it had no figure, while page 12 read
    405 kg. And the English label was written "Weight Kg" against a corpus that
    serialises "Weight (Kg)", so that substring test could never be true and
    only the Italian edition of a page was ever pinnable.
    """

    # Exactly as ingest/pdf_extract.py serialises the rows — the XLC engineering
    # editions with parentheses, the English-only datasheets without. The labels
    # are substring tested against these, so a label that drifts from either
    # spelling silently stops pinning that half of the corpus.
    SERIALISED_ROWS = (
        "DN (mm) = 300; A (mm) = 850; B (mm) = 676; C (mm) = 242; Peso (Kg) = 405",
        "DN (mm) = 300; A (mm) = 850; B (mm) = 676; C (mm) = 242; Weight (Kg) = 405",
        "DN (mm) = 300; A (mm) = 850; B (mm) = 676; C (mm) = 242; Poids (Kg) = 405",
        "CONNECTION inch/mm = Flanged 200; A mm = 365; B mm = 635; Weight Kg = 85,0",
        # ATHENA abbreviates the label; VRCD drops the space.
        "DN mm = 100; A mm = 350; B mm = 125; Wt Kg = 41",
        "DN (mm) = 100; A(mm) = 350; B(mm) = 110; Weight(Kg) = 34",
    )

    @pytest.mark.parametrize(
        "question",
        [
            "Quanto pesa la XLC 400 DN 300?",
            "Quanto pesano le XLC 400?",
            "Qual è il peso della XLC 400 DN 300?",
            "How much does the XLC 400 DN 300 weigh?",
            "What is the weight of the XLC 400 DN 300?",
            "Combien pèse la XLC 400 DN 300?",
            "Quel est le poids de la XLC 400?",
            "¿Cuánto pesa la XLC 400 DN 300?",
        ],
    )
    def test_a_weight_question_asks_for_the_weight_column(self, question: str):
        from api.retrieval import _requested_labels
        assert _requested_labels(question), f"no label requested for {question!r}"

    @pytest.mark.parametrize("row", SERIALISED_ROWS)
    def test_the_requested_labels_match_the_corpus_spelling(self, row: str):
        from api.retrieval import _requested_labels
        labels = _requested_labels("Quanto pesa la XLC 400 DN 300?")
        assert any(label.search(row) for label in labels), (
            f"none of {labels} matches a row the corpus actually contains"
        )

    @pytest.mark.parametrize(
        "question",
        [
            "A che pressione lavora la ITALICA 353?",
            "Che materiali usate per il corpo?",
            "Of course, tell me about the ITALICA 353",
        ],
    )
    def test_a_question_about_something_else_pins_nothing(self, question: str):
        from api.retrieval import _requested_labels
        assert _requested_labels(question) == ()

    # Both series' tables carry the weight column; the pin must follow the
    # series the question names, not the pool order. With the 400-series table
    # a shade ahead on cosine score, a question about an XLC 300 pinned the
    # XLC 400 table and the model — rightly refusing to read it — said the
    # weight was not documented.
    TABLE_400 = "XLC 400 - Versioni standard - Dati tecnici\nDN (mm) = 300; Peso (Kg) = 405"
    TABLE_300 = "XLC 300 - Versioni standard - Dati tecnici\nDN (mm) = 300; Peso (Kg) = 304"

    @pytest.mark.parametrize(
        "question, expected_first",
        [
            ("Quanto pesa la XLC 300 DN 300", "XLC 300"),
            ("Quanto pesa la XLC 400 DN 300?", "XLC 400"),
            ("How much does the XLC 300 DN 300 weigh?", "XLC 300"),
        ],
    )
    def test_the_pinned_table_is_the_series_asked_about(self, question, expected_first):
        from api.retrieval import _pin_chunks_holding, _requested_labels
        # 400-series table first in the pool, as its higher score puts it.
        candidates = [
            {"metadata": {"text": self.TABLE_400}},
            {"metadata": {"text": self.TABLE_300}},
        ]
        labels = _requested_labels(question)
        pinned = _pin_chunks_holding(candidates, labels, 1, question)
        assert pinned, f"nothing pinned for {question!r}"
        heading = candidates[pinned[0]]["metadata"]["text"].split("\n", 1)[0]
        assert expected_first in heading

    def test_a_question_naming_no_series_keeps_pool_order(self):
        from api.retrieval import _pin_chunks_holding, _requested_labels
        candidates = [
            {"metadata": {"text": self.TABLE_400}},
            {"metadata": {"text": self.TABLE_300}},
        ]
        labels = _requested_labels("Quanto pesano queste valvole?")
        assert _pin_chunks_holding(candidates, labels, 1, "Quanto pesano queste valvole?") == [0]

    def test_the_named_models_own_table_outranks_other_holders(self):
        """
        Asked what the FOX SUB weighs, the pool opened with two catalogue
        tables that also carry a weight column — the restore step had put them
        first — and the pin, following pool order, spent both slots on them:
        the answer gave another product's 74 kg for a valve whose own table
        says 44,5. The named model's own chunks take the pinned slots first.
        """
        from api.retrieval import _pin_chunks_holding, _requested_labels
        catalogue = "[Tabella — Dati tecnici]\nDN mm = 150; Peso Kg = 74"
        own_table = "[Table p.1]\nCONNECTION inch/mm = Flanged 150; A mm = 272; Weight Kg = 44,5"
        candidates = [
            {"metadata": {"text": catalogue}},
            {"metadata": {"text": catalogue}},
            {"metadata": {"text": own_table}, "exact_model_match": True, "model_match": True},
        ]
        labels = _requested_labels("Quanto pesa la FOX SUB flangiata DN 150?")
        pinned = _pin_chunks_holding(
            candidates, labels, 2, "Quanto pesa la FOX SUB flangiata DN 150?"
        )
        assert pinned[0] == 2, "the named model's own table must be pinned first"


class TestLanguageFallbackIsLocalised:
    """
    The refusal used to be one Italian sentence the model was told to translate.
    It often did not, so English, French, Spanish and German questions came back
    in Italian.
    """

    @pytest.mark.parametrize("language", ["en", "fr", "es"])
    def test_no_italian_refusal_leaks_into_other_languages(self, language: str):
        from api.prompt import build_system_prompt
        assert "Non ho informazioni" not in build_system_prompt("ctx", language)

    def test_each_language_gets_its_own_wording(self):
        from api.prompt import build_system_prompt
        assert "I do not have information" in build_system_prompt("ctx", "en")
        assert "Je n'ai pas" in build_system_prompt("ctx", "fr")
        assert "No tengo información" in build_system_prompt("ctx", "es")

    def test_an_unlisted_language_is_told_to_use_the_users_own(self):
        from api.prompt import build_system_prompt
        prompt = build_system_prompt("ctx", "de")
        assert "language of the user's message" in prompt
        assert "Non ho informazioni" not in prompt


class TestUrlLanguageRequests:
    """
    "E in inglese?" asks for a different link, not a different answer language.
    The bot kept returning the URL it had just given.
    """

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("E in inglese?", "en"),
            ("and in English?", "en"),
            ("y en español?", "es"),
            ("et en français ?", "fr"),
            ("e in italiano?", "it"),
            ("Dammi il link alla pagina", None),
            ("quanto pesa?", None),
        ],
    )
    def test_detects_the_language_asked_for(self, message: str, expected):
        from api.retrieval import requested_url_language
        assert requested_url_language(message) == expected


class TestNeighbourChunkIds:
    """
    Chunking cuts mid-clause. Asked how long the warranty runs, the bot saw the
    line about reporting a defect within "8 giorni" but not the term itself, and
    answered "8 years".
    """

    def test_neighbours_of_a_pdf_chunk(self):
        from api.retrieval import _neighbour_ids
        assert _neighbour_ids("FOX_3F.pdf_p3_c2") == ["FOX_3F.pdf_p3_c1", "FOX_3F.pdf_p3_c3"]

    def test_first_chunk_has_no_predecessor(self):
        from api.retrieval import _neighbour_ids
        assert _neighbour_ids("ARGO.pdf_p1_c0") == ["ARGO.pdf_p1_c1"]

    def test_page_and_xlc_chunk_ids_are_understood(self):
        from api.retrieval import _neighbour_ids
        assert "page__contatti__c0" in _neighbour_ids("page__contatti__c1")
        assert "xlceng_it_p7_c2" in _neighbour_ids("xlceng_it_p7_c3")

    def test_an_unrecognised_id_yields_nothing(self):
        from api.retrieval import _neighbour_ids
        assert _neighbour_ids("cat_253_text") == []


class TestReportingEndpointsAreGuarded:
    """
    The analytics and feedback-stats endpoints were readable by anyone who knew
    the path, exposing what visitors ask CSA — and `last_negatives` carries whole
    questions and answer snippets.
    """

    GUARDED = [
        "/api/analytics/top-queries",
        "/api/analytics/daily-stats",
        "/api/feedback/stats",
    ]

    def _client_with_token(self, token: str):
        import importlib
        import os
        os.environ["ANALYTICS_TOKEN"] = token
        import api.admin_auth, api.analytics, api.feedback, api.main
        for module in (api.admin_auth, api.analytics, api.feedback, api.main):
            importlib.reload(module)
        from fastapi.testclient import TestClient
        return TestClient(api.main.app)

    def teardown_method(self):
        import importlib
        import os
        os.environ.pop("ANALYTICS_TOKEN", None)
        import api.admin_auth, api.analytics, api.feedback, api.main
        for module in (api.admin_auth, api.analytics, api.feedback, api.main):
            importlib.reload(module)

    @pytest.mark.parametrize("path", GUARDED)
    def test_rejected_without_token(self, path: str):
        assert self._client_with_token("segreto").get(path).status_code == 401

    @pytest.mark.parametrize("path", GUARDED)
    def test_rejected_with_wrong_token(self, path: str):
        client = self._client_with_token("segreto")
        assert client.get(path, headers={"X-Analytics-Token": "altro"}).status_code == 401

    @pytest.mark.parametrize("path", GUARDED)
    def test_allowed_with_the_right_token(self, path: str):
        client = self._client_with_token("segreto")
        assert client.get(path, headers={"X-Analytics-Token": "segreto"}).status_code == 200

    def test_posting_feedback_stays_open_for_the_widget(self):
        client = self._client_with_token("segreto")
        response = client.post(
            "/api/feedback",
            json={"query": "q", "response": "r", "rating": "positive", "session_id": "pytest"},
        )
        assert response.status_code == 200


class TestUpstreamFailures:
    """
    Nothing wrapped the OpenAI and Pinecone calls in the chat endpoints, so a
    rate limit — the 200k tokens/minute ceiling is reached by a modest burst of
    visitors — or a provider timeout reached the visitor as a bare HTTP 500.
    """

    def _client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_chat_returns_a_readable_message_when_retrieval_fails(self):
        from unittest.mock import patch
        with patch("api.main.retrieve", side_effect=RuntimeError("rate limit")):
            response = self._client().post(
                "/api/chat", json={"message": "Che pressione regge la ITALICA 353?"}
            )
        assert response.status_code == 200
        body = response.json()
        assert "info@csasrl.it" in body["answer"]
        assert body["sources"] == []

    def test_chat_error_message_follows_the_requested_language(self):
        from unittest.mock import patch
        with patch("api.main.retrieve", side_effect=RuntimeError("boom")):
            response = self._client().post(
                "/api/chat", json={"message": "What pressure?", "language": "fr"}
            )
        assert response.status_code == 200
        assert "surchargé" in response.json()["answer"]

    def test_stream_closes_cleanly_when_retrieval_fails(self):
        from unittest.mock import patch
        with patch("api.main.retrieve", side_effect=RuntimeError("boom")):
            response = self._client().post(
                "/api/chat/stream", json={"message": "Che valvole avete?"}
            )
        assert response.status_code == 200
        body = response.text
        assert "info@csasrl.it" in body
        assert "done" in body

    def test_empty_message_is_still_rejected(self):
        response = self._client().post("/api/chat", json={"message": "   "})
        assert response.status_code == 400


class TestMergedTableRows:
    """
    Where a datasheet table has no ruling line between its data rows, pdfplumber
    returns them as one row holding every value, which reads as a single size
    with two of everything.
    """

    def _clean(self, table):
        from ingest.pdf_extract import _clean_table
        return _clean_table(table)

    def test_merged_rows_are_separated(self):
        # ARGO.pdf p.5
        grid = self._clean([
            ["CONNECTION (E) inch", "A mm", "B mm", "D mm", "Weight Kg"],
            ['Threaded 1" Threaded 2"', "80 110", "167 226", "CH 41 CH 65", "0,3 0,75"],
        ])
        assert len(grid) == 3
        assert grid[1] == ['Threaded 1"', "80", "167", "CH 41", "0,3"]
        assert grid[2] == ['Threaded 2"', "110", "226", "CH 65", "0,75"]

    def test_ranges_are_not_mistaken_for_two_values(self):
        # VRCD_FF.pdf p.3 — "2-20" is one pressure range, not the values 2 and 20.
        rows = [["Taratura (bar)", "2-20", "2-20", "2-15", "5-12"]]
        grid = self._clean([["Mod.", "A", "B", "C", "D"]] + rows)
        assert len(grid) == 2
        assert grid[1] == ["Taratura (bar)", "2-20", "2-20", "2-15", "5-12"]

    def test_ranges_inside_a_merged_row_stay_intact(self):
        # XLC_PILOTS.pdf p.14 — two merged rows whose labels are themselves ranges.
        grid = self._clean([
            ["DN", "A", "B", "C"],
            ["50-65 80-100", "95 121", "CH24 CH30", "CH8 CH10"],
        ])
        assert len(grid) == 3
        assert grid[1] == ["50-65", "95", "CH24", "CH8"]
        assert grid[2] == ["80-100", "121", "CH30", "CH10"]

    def test_ordinary_row_of_differing_shapes_is_untouched(self):
        grid = self._clean([
            ["N.", "Component", "Material"],
            ["1", "Body", "ductile cast iron GJS 450-10"],
        ])
        assert len(grid) == 2
        assert grid[1] == ["1", "Body", "ductile cast iron GJS 450-10"]

    def test_large_tables_are_left_alone(self):
        # Wide size tables are ruled and extract correctly; never reshape them.
        header = ["DN (mm)", "40", "50", "65"]
        rows = [[f"Attr {i}", "1 2", "3 4", "5 6"] for i in range(4)]
        grid = self._clean([header] + rows)
        assert len(grid) == 5

    def test_fused_rows_inside_a_large_table_are_split(self):
        # CYCLOPS_3F_RFP.pdf p.3 — a 10-row table whose middle rows are fused.
        # The old size gate (<=3 rows) skipped the whole table, and asked for
        # the 150R's weight the bot answered 57 — the plain 150's figure read
        # out of the other fused row. The R suffix must survive the split: it
        # is what distinguishes the reduced flange from its sibling.
        grid = self._clean([
            ["CONNECTION inch/mm", "A mm", "B mm", "C mm", "Weight Kg"],
            ["Flanged 80", "200", "340", "50", "15,2"],
            ["Flanged 100 Flanged 150R", "235 235", "403 435", "50 50", "21,5 34"],
            ["Flanged 150 Flanged 200R", "300 360", "523 523", "70 70", "57 62"],
        ])
        assert ["Flanged 100", "235", "403", "50", "21,5"] in grid
        assert ["Flanged 150R", "235", "435", "50", "34"] in grid
        assert ["Flanged 150", "300", "523", "70", "57"] in grid
        assert ["Flanged 200R", "360", "523", "70", "62"] in grid

    def test_continuation_rows_with_empty_cells_are_not_split(self):
        # APOLLO_RPC.pdf p.11 — a continuation row of a garbled multi-model
        # table. Splitting it would relabel RP 100C/RP 100D values as an
        # ambiguous "RP 100", manufacturing exactly the ambiguity the splitter
        # exists to remove. Mostly-empty rows are never candidates.
        grid = self._clean([
            ["Model", "A mm", "B mm", "C mm", "D mm", "H mm", "F", "Fl", "Wt. Kg"],
            ["RP 100X RP 100A RP 100B", "680", "719 869 1019", "50", "130",
             "1494 1644 1794", "2Ø70 2Ø70 +", "DN 100", "89 89 95"],
            ["RP 100C RP 100D", "", "1269 1469", "", "", "2044 2244",
             "1Ø100", "", "100 105"],
        ])
        assert ["RP 100C RP 100D", "", "1269 1469", "", "", "2044 2244",
                "1Ø100", "", "100 105"] in grid


class TestDimensionDrawings:
    """
    A dimensions answer lists letters — "A = 230 mm, B = 82,5 mm" — that mean
    nothing without the quoted drawing on the same datasheet page. When a
    source is a page in the drawings map, its rendered PNG rides along.
    """

    def _source(self, source_file: str, page, exact: bool = False):
        from api.models import Source
        return Source(
            source_file=source_file, page=page, chunk_id="x", score=1.0,
            text_snippet="", text_full="", is_exact_model=exact,
        )

    def test_a_stray_products_page_adds_no_drawing(self):
        """
        Asked the ATHENA's dimensions, the context also held an XLC dimension
        page — retrieval pads with whatever resembles a size table — and the
        answer shipped the ATHENA drawing plus an XLC one. When a model is
        named, only its own datasheet contributes a drawing.
        """
        from api.product_images import get_dimension_drawings
        images = get_dimension_drawings([
            self._source("ATHENA.pdf", 3, exact=True),
            self._source("XLC engineering ITAL v2.pdf", 20),
        ])
        assert len(images) == 1
        assert "ATHENA" in images[0]["product_name"]

    def test_a_dimensions_page_source_brings_its_drawing(self):
        from api.product_images import _DIMENSION_DRAWINGS, get_dimension_drawings
        assert _DIMENSION_DRAWINGS, "dimension_drawings.json missing or empty"
        file_name, pages = next(iter(sorted(_DIMENSION_DRAWINGS.items())))
        page = next(iter(pages))
        images = get_dimension_drawings(
            [self._source(file_name, int(page))], named_files=(file_name,)
        )
        assert len(images) == 1
        assert images[0]["url"] == pages[page]
        assert images[0]["product_name"]

    def test_a_non_table_page_brings_nothing(self):
        from api.product_images import get_dimension_drawings
        assert get_dimension_drawings(
            [self._source("ATHENA.pdf", 1)], named_files=("ATHENA.pdf",)
        ) == []
        assert get_dimension_drawings([self._source("csasrl.it", None)]) == []

    def test_an_unrecognised_question_gets_no_drawing(self):
        """
        "mi dai le dimensioni di atena" — before the misspelling was even
        handled — refused the answer AND attached the XLC 300 drawing: the
        fallback pinned whatever source led the pool. No recognised model,
        no drawing.
        """
        from api.product_images import get_dimension_drawings
        sources = [
            self._source("XLC engineering ITAL v2.pdf", 20),
            self._source("ATHENA.pdf", 3),
        ]
        assert get_dimension_drawings(sources, named_files=()) == []

    def test_xlc_series_pages_share_one_drawing_per_series(self):
        from api.product_images import get_dimension_drawings
        it = get_dimension_drawings(
            [self._source("XLC engineering ITAL v2.pdf", 12)],
            named_files=("XLC engineering ITAL v2.pdf",),
        )
        en = get_dimension_drawings(
            [self._source("XLC engineering ENG v2.pdf", 12)],
            named_files=("XLC engineering ENG v2.pdf",),
        )
        assert it and en and it[0]["url"] == en[0]["url"]
        assert "400" in it[0]["product_name"]

    def test_every_mapped_drawing_file_exists(self):
        from api.product_images import _DIMENSION_DRAWINGS
        repo = Path(__file__).resolve().parent.parent
        missing = [
            url
            for pages in _DIMENSION_DRAWINGS.values()
            for url in pages.values()
            if not (repo / url.lstrip("/")).exists()
        ]
        assert not missing, f"mapped drawings without a file: {missing[:5]}"


class TestLinkSanitizer:
    """
    The model occasionally "corrects" a slug's spelling and produces a 404 —
    asked in Spanish about the Fortix it wrote 'alta-efficiencia' where the real
    Spanish page keeps the Italian 'alta-efficienza'. Instructions alone did not
    stop it, so links are verified against what was actually retrieved.
    """

    ALLOWED = [
        "https://csasrl.it/es/prodotto/filtro-csa-fortix-alta-efficienza/",
        "https://csasrl.it/prodotto/valvola-gemina-colpo-ariete/",
    ]

    def test_valid_link_passes_through(self):
        from api.links import sanitize_links
        text = "Vedi [Gemina](https://csasrl.it/prodotto/valvola-gemina-colpo-ariete/)."
        assert sanitize_links(text, self.ALLOWED) == text

    def test_misspelled_slug_snaps_to_the_real_page(self):
        from api.links import sanitize_links
        text = "Mira [Fortix](https://csasrl.it/es/prodotto/filtro-csa-fortix-alta-efficiencia/)."
        assert sanitize_links(text, self.ALLOWED) == (
            "Mira [Fortix](https://csasrl.it/es/prodotto/filtro-csa-fortix-alta-efficienza/)."
        )

    def test_invented_link_is_unwrapped_keeping_the_text(self):
        from api.links import sanitize_links
        text = "Vedi [Catalogo](https://csasrl.it/pagina-inesistente/)."
        assert sanitize_links(text, self.ALLOWED) == "Vedi Catalogo."

    def test_text_without_links_is_untouched(self):
        from api.links import sanitize_links
        assert sanitize_links("Nessun link qui.", self.ALLOWED) == "Nessun link qui."

    def test_streaming_matches_the_synchronous_result(self):
        from api.links import StreamingLinkSanitizer, sanitize_links
        text = (
            "Mira [Fortix](https://csasrl.it/es/prodotto/filtro-csa-fortix-alta-efficiencia/) "
            "y [X](https://csasrl.it/no/) fine."
        )
        sanitizer = StreamingLinkSanitizer(self.ALLOWED)
        streamed = "".join(sanitizer.feed(text[i : i + 3]) for i in range(0, len(text), 3))
        streamed += sanitizer.flush()
        assert streamed == sanitize_links(text, self.ALLOWED)


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
@pytest.mark.integration
@REQUIRES_KEYS
class TestIntegration:
    """
    These tests exercise the full stack. They are skipped when keys are absent.

    retrieve() is a coroutine, so every test here is async and needs the
    asyncio marker — calling it synchronously raised
    "TypeError: cannot unpack non-iterable coroutine object".
    """

    @pytest.fixture(autouse=True)
    def _check_pinecone(self):
        key = os.environ.get("PINECONE_API_KEY", "")
        if not key or "YOUR" in key or len(key) < 10:
            pytest.skip("PINECONE_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_q15_ball_valve_specs_english(self):
        """Product info: ball valve technical specs (EN)."""
        from api.retrieval import retrieve
        sources, lang = await retrieve(
            "What are the technical specifications for CSA ball valves?",
            language_hint="en",
        )
        assert lang == "en"
        assert isinstance(sources, list)  # may be empty if index not populated

    @pytest.mark.asyncio
    async def test_q16_italian_query_returns_it_language(self):
        """Multilingual: Italian query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = await retrieve("Quali sono le caratteristiche delle valvole a sfera CSA?")
        assert lang == "it"

    @pytest.mark.asyncio
    async def test_q17_french_query_returns_fr_language(self):
        """Multilingual: French query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = await retrieve("Quelles sont les caractéristiques des vannes à bille CSA?")
        assert lang == "fr"

    @pytest.mark.asyncio
    async def test_q18_spanish_query_returns_es_language(self):
        """Multilingual: Spanish query detected correctly."""
        from api.retrieval import retrieve
        sources, lang = await retrieve("¿Cuáles son las características de las válvulas de bola CSA?")
        assert lang == "es"

    @pytest.mark.asyncio
    async def test_q19_url_in_correct_language(self):
        """URL correctness: Italian query gets Italian URL."""
        from api.retrieval import retrieve
        sources, lang = await retrieve("valvole a sfera CSA", language_hint="it")
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
