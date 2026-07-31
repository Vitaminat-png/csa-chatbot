"""
ingest/pdf_ingest.py
--------------------
Ingests English PDF documents from the docs/ folder into Pinecone.

Steps:
1. Extract text page-by-page with pdfplumber.
2. Split into chunks of ~500 tokens with 50-token overlap (tiktoken).
3. Embed each chunk with OpenAI text-embedding-3-small (dim=1536).
4. Upsert vectors to Pinecone with metadata:
   - source_file: filename
   - page: page number (1-based)
   - chunk_id: "filename_p{page}_c{chunk_index}"
   - text: the raw chunk text (stored for retrieval display)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Generator

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "csa-chatbot")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 20  # reduced for free-tier timeout tolerance

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Chunking and page extraction live in pdf_extract so every ingest script shares
# them. Re-exported here because tests and site_crawler import them from this
# module.
from ingest.pdf_extract import (  # noqa: E402  (kept next to the config above)
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _decode,
    _enc,
    _tokenize,
    chunk_text,
    page_units,
)


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------
def get_or_create_index(pc: Pinecone) -> object:
    """Return the Pinecone index, creating it (serverless) if absent."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"[pinecone] Creating index '{PINECONE_INDEX_NAME}' …")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("[pinecone] Index ready — waiting 15s for warm-up …")
        time.sleep(15)
    return pc.Index(PINECONE_INDEX_NAME)


def _batch(iterable: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts; returns list of float vectors."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------
def delete_stale_chunks(index, filename: str, keep: set[str]) -> int:
    """
    Drop vectors of *filename* that this run did not rewrite.

    Chunk ids encode page and position, so a document that now yields fewer
    chunks on a page leaves the surplus behind — stale text that keeps being
    retrieved. This matters after the switch to table-aware extraction, which
    changes almost every document's chunk layout.
    """
    stale = [
        vid
        for vid in _ids_with_prefix(index, f"{filename}_p")
        if vid not in keep
    ]
    for i in range(0, len(stale), 100):
        index.delete(ids=stale[i : i + 100])
    return len(stale)


def _ids_with_prefix(index, prefix: str) -> list[str]:
    """Collect every vector id under *prefix* via the paginated list API."""
    ids: list[str] = []
    try:
        for page in index.list(prefix=prefix, limit=100):
            if hasattr(page, "vectors"):
                ids += [getattr(v, "id", str(v)) for v in page.vectors]
            elif isinstance(page, list):
                ids += [getattr(v, "id", str(v)) for v in page]
            elif isinstance(page, str):
                ids.append(page)
    except Exception as exc:
        print(f"  [warn] could not list '{prefix}' vectors ({exc})")
    return ids


def ingest_pdf(pdf_path: Path, oai: OpenAI, index: object) -> int:
    """Ingest one PDF. Returns number of vectors upserted."""
    filename = pdf_path.name
    print(f"[ingest] Processing '{filename}' …")

    records: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # page_units serialises tables and strips chart-axis noise; plain
            # extract_text() flattened the DN/Kv and dimension tables into a
            # single line, severing every value from its size.
            for chunk_idx, chunk in enumerate(page_units(page, page_num)):
                chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
                records.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            "source_file": filename,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_idx,
                            "text": chunk,  # stored for display in retrieval
                        },
                    }
                )

    total = 0
    for batch in _batch(records, BATCH_SIZE):
        texts = [r["text"] for r in batch]
        embeddings = embed_texts(oai, texts)
        vectors = [
            {
                "id": r["id"],
                "values": emb,
                "metadata": r["metadata"],
            }
            for r, emb in zip(batch, embeddings)
        ]
        # Retry upsert up to 3 times on transient errors
        for attempt in range(3):
            try:
                index.upsert(vectors=vectors, timeout=120)
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"  [warn] upsert attempt {attempt+1} failed ({exc}), retrying in {wait}s …")
                    time.sleep(wait)
                else:
                    raise
        total += len(vectors)
        print(f"  upserted {total}/{len(records)} chunks …")

    removed = delete_stale_chunks(index, filename, {r["id"] for r in records})
    suffix = f", {removed} stale removed" if removed else ""
    print(f"[ingest] '{filename}' done — {total} chunks indexed{suffix}.")
    return total


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> None:
    from ingest.build_model_registry import CATALOGUE_MARKERS

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    # Whole catalogues belong to ingest/catalog_ingest.py, which indexes them
    # into the 'catalog' namespace with section and product-family metadata.
    # Processing them here as well produced a second, poorer copy of the same
    # 400-plus pages in the default namespace.
    catalogues = [p for p in pdf_files if any(m in p.stem.lower() for m in CATALOGUE_MARKERS)]
    pdf_files = [p for p in pdf_files if p not in catalogues]
    for path in catalogues:
        print(f"[ingest] Skipping '{path.name}' — handled by catalog_ingest.py")

    if not pdf_files:
        print(f"[ingest] No PDF files found in {DOCS_DIR}. Add PDFs and re-run.")
        return

    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    grand_total = 0
    failed: list[str] = []
    for pdf_path in pdf_files:
        try:
            grand_total += ingest_pdf(pdf_path, oai, index)
        except Exception as exc:
            print(f"[error] Failed to ingest '{pdf_path.name}': {exc}")
            failed.append(pdf_path.name)

    if failed:
        print(f"\n[ingest] {len(failed)} file(s) failed:")
        for name in failed:
            print(f"  - {name}")
    print(f"\n[ingest] All done. Total vectors upserted: {grand_total}")


if __name__ == "__main__":
    main()
