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
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
BATCH_SIZE = 20  # reduced for free-tier timeout tolerance

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------------------
# Tokeniser (cl100k_base is used by text-embedding-3-small)
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def _decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping token-based chunks."""
    tokens = _tokenize(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


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
def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), …] for each non-empty page."""
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append((i, text))
    return pages


def ingest_pdf(pdf_path: Path, oai: OpenAI, index: object) -> int:
    """Ingest one PDF. Returns number of vectors upserted."""
    filename = pdf_path.name
    print(f"[ingest] Processing '{filename}' …")

    pages = extract_pages(pdf_path)
    records: list[dict] = []

    for page_num, page_text in pages:
        chunks = chunk_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
            records.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source_file": filename,
                        "page": page_num,
                        "chunk_id": chunk_id,
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

    print(f"[ingest] '{filename}' done — {total} chunks indexed.")
    return total


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> None:
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
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
