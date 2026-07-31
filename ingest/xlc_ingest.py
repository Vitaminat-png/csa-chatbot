"""
ingest/xlc_ingest.py
--------------------
Ingests the XLC engineering documents ("xlc engeniering/") into Pinecone.

These four PDFs (it/en/fr/es) are the current authority on the XLC 300/400
range — sizes, Kv values, flow rates, materials — and supersede the older
per-model XLC datasheets in docs/.

Two things make this different from ingest/pdf_ingest.py:

1. Tables are serialised, not flattened. pdfplumber's extract_text() collapses
   a dimension table into one line ("DN (mm) 40 50 65 ... Kv (m3/h) 40,6 40,6
   68 ...") which destroys the DN-to-value association and, on the XLC sizing
   pages, drops the last column outright — the DN 800 sizes were missing
   entirely. Sizes are exactly what these documents are consulted for, so each
   column is emitted as its own labelled record.

2. Chunks carry `lang`, `doc_group` and `doc_priority` metadata so retrieval can
   prefer the reader's language, collapse the four translations of one page into
   a single context slot, and rank this set above the superseded datasheets.

Run standalone:
    python -m ingest.xlc_ingest
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from ingest.build_model_registry import dedupe_by_content
from ingest.pdf_extract import page_units
from ingest.pdf_ingest import (
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    _batch,
    BATCH_SIZE,
    embed_texts,
    get_or_create_index,
)

load_dotenv()

XLC_DIR = Path(__file__).resolve().parent.parent / "xlc engeniering"

# Chunks from this set share a group id so retrieval can recognise the four
# language editions of the same page as one another's translations.
DOC_GROUP = "xlc_engineering"
PRODUCT_FAMILY = "XLC"

# Filename token -> BCP-47 code.
LANG_BY_MARKER: dict[str, str] = {
    "ITAL": "it",
    "ITA": "it",
    "ENG": "en",
    "FRA": "fr",
    "SPA": "es",
}


def detect_language(filename: str) -> str | None:
    """
    Return the BCP-47 code encoded in *filename*, or None if absent.

    Matching is on whole tokens, not substrings: 'XLC engineering FRA v2.pdf'
    contains 'ENG' inside 'engineering', which a substring test reads as English
    and then files the French edition under the English vector ids.
    """
    tokens = {t.upper() for t in re.split(r"[^A-Za-z0-9]+", filename) if t}
    for marker, code in LANG_BY_MARKER.items():
        if marker in tokens:
            return code
    return None


# ---------------------------------------------------------------------------
# Table serialisation
# ---------------------------------------------------------------------------
# Page headings name the series and variant, e.g. "XLC 400 - Versione standard -
# Dati tecnici". The non-digit tail stops the match before the technical data.
_HEADING = re.compile(r"XLC\s*\d{3}(?:/\d{3})?[^\d]{0,60}")


def page_heading(page) -> str:
    """
    Return the series/variant heading printed on the page.

    Every chunk is prefixed with it because the document covers two ranges: the
    full-bore XLC 400 on pages 6-13 and the reduced-bore XLC 300 from page 14.
    A bare "DN (mm) = 350; Kv …" table does not say which range it belongs to,
    and a question about the XLC 400 was answered with XLC 300 sizes.
    """
    text = re.sub(r"\s+", " ", page.extract_text() or "")
    match = _HEADING.search(text)
    if not match:
        return ""

    # Keep the series plus its descriptor segments; the regex tail runs on into
    # whatever text follows the heading, so drop anything past the third dash.
    parts = [p.strip() for p in match.group(0).split(" - ") if p.strip()]
    heading = " - ".join(parts[:3])
    # Trim a trailing table column label that followed the heading on the page.
    return re.sub(r"\s+DN\s*(?:\([^)]*\))?$", "", heading).strip()


def ingest_pdf(path: Path, lang: str, oai: OpenAI, index) -> int:
    """Ingest one language edition. Returns the number of vectors upserted."""
    filename = path.name
    print(f"[xlc] {filename} (lang={lang}) …")

    records: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            units = page_units(page, page_num, heading_fn=page_heading)
            for chunk_idx, chunk in enumerate(units):
                chunk_id = f"xlceng_{lang}_p{page_num}_c{chunk_idx}"
                records.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            "source_file": filename,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_idx,
                            "text": chunk,
                            "lang": lang,
                            "doc_group": DOC_GROUP,
                            "doc_priority": 1,
                            "product_family": PRODUCT_FAMILY,
                            "valve_model": PRODUCT_FAMILY,
                        },
                    }
                )

    total = 0
    for batch in _batch(records, BATCH_SIZE):
        embeddings = embed_texts(oai, [r["text"] for r in batch])
        vectors = [
            {"id": r["id"], "values": emb, "metadata": r["metadata"]}
            for r, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors, timeout=120)
        total += len(vectors)
        print(f"    upserted {total}/{len(records)} …")

    print(f"[xlc] {filename} done — {total} chunks.")
    return total


ID_PREFIX = "xlceng_"


def delete_existing(index) -> int:
    """
    Remove every vector from a previous run of this script.

    Re-running without this leaves orphans behind whenever the chunk count of a
    page changes, because ids are derived from page and chunk position — a
    shorter run simply stops overwriting partway through.
    """
    ids: list[str] = []
    try:
        for page in index.list(prefix=ID_PREFIX, limit=100):
            if hasattr(page, "vectors"):
                ids += [getattr(v, "id", str(v)) for v in page.vectors]
            elif isinstance(page, list):
                ids += [getattr(v, "id", str(v)) for v in page]
            elif isinstance(page, str):
                ids.append(page)
    except Exception as exc:
        print(f"[xlc] [warn] could not list existing vectors ({exc}); skipping cleanup")
        return 0

    if not ids:
        return 0

    for i in range(0, len(ids), 100):
        index.delete(ids=ids[i : i + 100])
    print(f"[xlc] Removed {len(ids)} vectors from the previous run.")
    return len(ids)


def main() -> None:
    if not XLC_DIR.exists():
        print(f"[xlc] Folder not found: {XLC_DIR}")
        return

    # The folder ships each language twice under different names; the 'v2' and
    # plain files are byte-identical, so only one copy of each is ingested.
    paths = dedupe_by_content(list(XLC_DIR.glob("*.pdf")))
    print(f"[xlc] {len(paths)} unique documents after de-duplication.")

    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    delete_existing(index)

    grand_total = 0
    for path in paths:
        lang = detect_language(path.name)
        if lang is None:
            print(f"[xlc] [skip] cannot determine language from '{path.name}'")
            continue
        grand_total += ingest_pdf(path, lang, oai, index)

    print(f"\n[xlc] Done. {grand_total} vectors upserted.")


if __name__ == "__main__":
    main()
