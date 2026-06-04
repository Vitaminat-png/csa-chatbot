"""
ingest/catalog_ingest.py
------------------------
Specialized ingestion for the CSA full product catalog PDF.

Differences from pdf_ingest.py:
- Uses pdfplumber to extract BOTH text AND tables per page
- Tables are formatted as pipe-separated markdown for precise retrieval
- Smaller chunks: 300 tokens, 100 overlap (preserves table data)
- Rich metadata: page_number, section_name, product_family, valve_model
- Upserts to namespace 'catalog' (separate from datasheets)
- Logs progress every 20 pages
- On per-page failure: logs and continues

Usage:
    cd csa-chatbot
    python -m ingest.catalog_ingest
  or
    python ingest/catalog_ingest.py
"""

from __future__ import annotations

import os
import re
import time
import hashlib
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
CHUNK_SIZE = 300        # smaller = more precise retrieval for technical specs
CHUNK_OVERLAP = 100     # generous overlap so table rows don't get split across chunks
BATCH_SIZE = 20         # free-tier safe
NAMESPACE = "catalog"   # keep separate from datasheets namespace

CATALOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "Catalogo Ita 05-26.pdf"

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def _decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ---------------------------------------------------------------------------
# Section / product detection heuristics (Italian catalog)
# ---------------------------------------------------------------------------
# Detect top-level section from page header text
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bAIR\s+VALVE", "Air Valves"),
    (r"\bSFIATO\b|\bSFIATI\b", "Sfiati"),
    (r"\bBUTTERFLY", "Butterfly Valves"),
    (r"\bFARFALL", "Valvole a Farfalla"),
    (r"\bGATE\s+VALVE|\bSARAC", "Gate Valves / Saracinesche"),
    (r"\bCHECK\s+VALVE|\bVALVOLA\s+DI\s+RITENU", "Check Valves"),
    (r"\bCONTROL\s+VALVE|\bVALVOLA\s+DI\s+REGOLA", "Control Valves"),
    (r"\bBALL\s+VALVE|\bVALVOLA\s+A\s+SFERA", "Ball Valves"),
    (r"\bPRESSURE\s+REDUC|\bRIDUTTRICE", "Pressure Reducing"),
    (r"\bRELIEF|\bSICUREZZA", "Relief / Safety Valves"),
    (r"\bSURGE|\bARIETE\b", "Surge Protection"),
    (r"\bFILTR", "Filters"),
    (r"\bACCESSOR", "Accessories"),
    (r"\bINDIC|\bINDEX\b|\bINDICE\b", "Index"),
]

# Detect valve model names (uppercase alphanumeric tokens ≥3 chars that look like model codes)
_MODEL_RE = re.compile(
    r"\b(FOX|LYNX|EAGLE|HAWK|ARGO|ITALICA|ATLAS|HERMES|TITAN|ZEUS|ORION|"
    r"XLC|CORA|LINDA|ARES|PRIMUS|DELTA|SIGMA|OMEGA|VEGA|LYRA|CETUS|"
    r"IRIS|DIVA|NOVA|LUNA|SOLARIS|AURORA|POLLUX|CASTOR|"
    r"[A-Z]{2,6}-\d{3,4}[A-Z]?)\b"
)

# Detect product family from text
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"\bFOX\b", "FOX"),
    (r"\bLYNX\b", "LYNX"),
    (r"\bEAGLE\b", "EAGLE"),
    (r"\bHAWK\b", "HAWK"),
    (r"\bARGO\b", "ARGO"),
    (r"\bITALICA\b", "ITALICA"),
    (r"\bATLAS\b", "ATLAS"),
    (r"\bHERMES\b", "HERMES"),
    (r"\bXLC\b", "XLC"),
    (r"\bCORA\b", "CORA"),
    (r"\bLINDA\b", "LINDA"),
]


def detect_section(text: str) -> str:
    upper = text[:500].upper()  # only check top of page
    for pattern, name in _SECTION_PATTERNS:
        if re.search(pattern, upper):
            return name
    return "General"


def detect_product_family(text: str) -> str:
    upper = text.upper()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, upper):
            return family
    return ""


def detect_valve_models(text: str) -> str:
    models = _MODEL_RE.findall(text.upper())
    # deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return ", ".join(unique[:5])  # cap at 5 to keep metadata small


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------
def format_table(table: list[list[str | None]], page_context: str) -> str:
    """
    Convert pdfplumber table (list of rows, each a list of cells) into
    a markdown pipe-delimited string prefixed with the page context.
    """
    if not table:
        return ""

    # Clean cells
    def clean(cell: str | None) -> str:
        if cell is None:
            return ""
        return re.sub(r"\s+", " ", str(cell)).strip()

    cleaned_rows = [[clean(c) for c in row] for row in table]
    # Remove entirely empty rows
    cleaned_rows = [row for row in cleaned_rows if any(c for c in row)]
    if not cleaned_rows:
        return ""

    # First row as header
    header = " | ".join(cleaned_rows[0])
    separator = " | ".join(["---"] * len(cleaned_rows[0]))
    body_rows = [" | ".join(row) for row in cleaned_rows[1:]]

    table_md = "\n".join([header, separator] + body_rows)

    # Prefix with page context for traceability
    if page_context:
        return f"[Tabella — {page_context}]\n{table_md}"
    return table_md


# ---------------------------------------------------------------------------
# Page extraction (text + tables)
# ---------------------------------------------------------------------------
def extract_page_content(page, page_num: int) -> list[dict]:
    """
    Extract content from a single pdfplumber page.
    Returns a list of content blocks: {'type': 'text'|'table', 'content': str}
    """
    blocks: list[dict] = []

    # 1. Extract plain text
    text = page.extract_text() or ""
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        blocks.append({"type": "text", "content": text})

    # 2. Extract tables
    try:
        tables = page.extract_tables()
        if tables:
            # Use the first text block as context prefix for the table
            context = text[:120] if text else f"pagina {page_num}"
            for table in tables:
                table_str = format_table(table, context)
                if table_str and len(table_str) > 30:  # skip trivially small tables
                    blocks.append({"type": "table", "content": table_str})
    except Exception as exc:
        print(f"  [warn] table extraction failed page {page_num}: {exc}")

    return blocks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
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
def get_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"[pinecone] Creating index '{PINECONE_INDEX_NAME}' …")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("[pinecone] Index ready — waiting 15s …")
        time.sleep(15)
    return pc.Index(PINECONE_INDEX_NAME)


def _batch(items: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------
def upsert_batch(index, vectors: list[dict], namespace: str) -> None:
    for attempt in range(3):
        try:
            index.upsert(vectors=vectors, namespace=namespace, timeout=120)
            return
        except Exception as exc:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  [warn] upsert attempt {attempt+1} failed ({exc}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------
def ingest_catalog(pdf_path: Path, oai: OpenAI, index) -> int:
    print(f"\n[catalog_ingest] Opening '{pdf_path.name}' …")
    filename = pdf_path.name

    all_records: list[dict] = []
    failed_pages: list[int] = []
    current_section = "General"
    current_family = ""

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[catalog_ingest] Total pages: {total_pages}")

        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            try:
                blocks = extract_page_content(page, page_num)

                # Update running section/family from combined page text
                full_text = " ".join(b["content"] for b in blocks if b["type"] == "text")
                if full_text:
                    sec = detect_section(full_text)
                    if sec != "General":
                        current_section = sec
                    fam = detect_product_family(full_text)
                    if fam:
                        current_family = fam

                valve_models = detect_valve_models(full_text) if full_text else ""

                for block in blocks:
                    content = block["content"]
                    content_type = block["type"]
                    chunks = chunk_text(content)

                    for chunk_idx, chunk in enumerate(chunks):
                        # Build a stable ID using a hash of content
                        raw_id = f"cat_{page_num}_{content_type}_{chunk_idx}"
                        chunk_id = raw_id  # keep readable

                        all_records.append({
                            "id": chunk_id,
                            "text": chunk,
                            "metadata": {
                                "source_file": filename,
                                "page": page_num,
                                "chunk_id": chunk_id,
                                "content_type": content_type,   # "text" or "table"
                                "section_name": current_section,
                                "product_family": current_family,
                                "valve_model": valve_models,
                                "namespace": NAMESPACE,
                                "text": chunk,  # stored for display
                            },
                        })

            except Exception as exc:
                print(f"  [error] page {page_num} failed: {exc}")
                failed_pages.append(page_num)
                continue

            # Progress log every 20 pages
            if page_num % 20 == 0 or page_num == total_pages:
                print(f"  [progress] page {page_num}/{total_pages} — records so far: {len(all_records)}")

    print(f"\n[catalog_ingest] Extraction done. {len(all_records)} records from {total_pages} pages.")
    if failed_pages:
        print(f"  Failed pages: {failed_pages}")

    # --- Embedding + upsert in batches ---
    total_upserted = 0
    batch_num = 0

    for batch_records in _batch(all_records, BATCH_SIZE):
        batch_num += 1
        texts = [r["text"] for r in batch_records]
        try:
            embeddings = embed_texts(oai, texts)
        except Exception as exc:
            print(f"  [error] embedding batch {batch_num} failed: {exc} — skipping {len(batch_records)} records")
            continue

        vectors = [
            {"id": r["id"], "values": emb, "metadata": r["metadata"]}
            for r, emb in zip(batch_records, embeddings)
        ]

        try:
            upsert_batch(index, vectors, NAMESPACE)
        except Exception as exc:
            print(f"  [error] upsert batch {batch_num} failed permanently: {exc} — skipping")
            continue

        total_upserted += len(vectors)
        if batch_num % 20 == 0:
            print(f"  [upsert] {total_upserted}/{len(all_records)} vectors upserted …")

    print(f"\n[catalog_ingest] DONE. {total_upserted} vectors upserted to namespace '{NAMESPACE}'.")
    if failed_pages:
        print(f"  Pages with errors: {sorted(failed_pages)}")
    return total_upserted


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "Quali sono le differenze tra FOX e LYNX?",
    "Qual e il Kv della XLC-310 DN80?",
    "Quali materiali usa la valvola ARGO?",
    "Quanti modelli di sfiati ha CSA?",
    "A che pressione lavora la ITALICA 353?",
]


def run_test_queries(oai: OpenAI, index) -> None:
    print("\n" + "=" * 60)
    print("RUNNING TEST QUERIES (namespace='catalog')")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\nQ: {query}")
        try:
            emb = oai.embeddings.create(model=EMBED_MODEL, input=[query])
            vec = emb.data[0].embedding

            result = index.query(
                vector=vec,
                top_k=3,
                include_metadata=True,
                namespace=NAMESPACE,
            )
            matches = result.get("matches", [])
            if not matches:
                print("  -> No results found")
            else:
                for i, m in enumerate(matches, 1):
                    score = round(m.get("score", 0), 4)
                    meta = m.get("metadata", {})
                    snippet = meta.get("text", "")[:150].replace("\n", " ")
                    page = meta.get("page", "?")
                    section = meta.get("section_name", "")
                    ctype = meta.get("content_type", "")
                    print(f"  [{i}] score={score} page={page} [{ctype}] section={section}")
                    print(f"       {snippet}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"[error] Catalog PDF not found at: {CATALOG_PATH}")
        print("  Copy it to the docs/ folder and retry.")
        return

    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_index(pc)

    start = time.time()
    total = ingest_catalog(CATALOG_PATH, oai, index)
    elapsed = time.time() - start

    print(f"\n[summary] {total} vectors ingested in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    run_test_queries(oai, index)


if __name__ == "__main__":
    main()
