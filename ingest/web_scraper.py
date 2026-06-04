"""
ingest/web_scraper.py
---------------------
Scrapes csasrl.it sitemaps to build a complete URL map with language variants.

Steps:
1. Fetch /sitemap.xml (index sitemap) → find product-sitemap.xml / page-sitemap.xml
2. Parse each child sitemap for <loc> + <xhtml:link> hreflang entries.
3. Build url_map.json:  { canonical_url: { "it": "...", "en": "...", "fr": "...", "es": "..." } }
4. Upsert page title/description embeddings to Pinecone so the chatbot can
   surface correct language-specific links at query time.

Run standalone:
    python -m ingest.web_scraper
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
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

BASE_URL = "https://csasrl.it"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "url_map.json"

SUPPORTED_LANGS = {"it", "en", "fr", "es"}

# ---------------------------------------------------------------------------
# URL blocklist — paths never useful to ingest (mirrors retrieval.py filter)
# ---------------------------------------------------------------------------
BLOCKED_URL_PATTERNS: list[str] = [
    # E-commerce / account pages
    "/shop/",
    "/cart/",
    "/checkout/",
    "/my-account/",
    "/carrello/",
    "/pagamento/",
    "/mio-account/",
    "/mon-compte/",
    "/mi-cuenta/",
    "/boutique/",
    "/panier/",
    # Membership / login utility slugs
    "/account-2",
    "/edit-profile-2",
    "/logout-2",
    "/members-2",
    "/user-2",
    "/password-reset-2",
    "/password-reset-3",
    "/register-2",
    "/register-3",
    "/register-4",
]


def _is_blocked(url: str) -> bool:
    """Return True if *url* should be excluded from ingestion."""
    lower = url.lower()
    return any(pattern in lower for pattern in BLOCKED_URL_PATTERNS)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch(url: str, client: httpx.Client, retries: int = 3) -> Optional[str]:
    """GET *url* and return text content, or None on failure."""
    for attempt in range(retries):
        try:
            r = client.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except httpx.HTTPError as exc:
            print(f"  [warn] {url} attempt {attempt+1} failed: {exc}")
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------
def parse_sitemap_index(xml_text: str, base_url: str) -> list[str]:
    """Return child sitemap URLs from a sitemap index document."""
    soup = BeautifulSoup(xml_text, "xml")
    locs = soup.find_all("loc")
    urls = []
    for loc in locs:
        href = loc.get_text(strip=True)
        if href:
            urls.append(href if href.startswith("http") else urljoin(base_url, href))
    return urls


def parse_url_set(xml_text: str) -> list[dict]:
    """
    Parse a <urlset> sitemap.
    Returns list of dicts:
      {
        "loc": "https://...",
        "langs": {"it": "...", "en": "...", "fr": "...", "es": "..."}
      }
    Each dict may also contain "lastmod" if present.
    """
    soup = BeautifulSoup(xml_text, "xml")
    entries = []
    for url_tag in soup.find_all("url"):
        loc_tag = url_tag.find("loc")
        if not loc_tag:
            continue
        loc = loc_tag.get_text(strip=True)

        # hreflang links live in <xhtml:link> elements
        langs: dict[str, str] = {}
        for link_tag in url_tag.find_all("link"):
            hreflang = link_tag.get("hreflang", "")
            href = link_tag.get("href", "")
            # normalise: "en-US" -> "en", "fr-FR" -> "fr", etc.
            lang_code = hreflang.split("-")[0].lower()
            if lang_code in SUPPORTED_LANGS and href:
                langs[lang_code] = href

        # Fallback: if no hreflang but URL contains a language path segment
        if not langs:
            parsed = urlparse(loc)
            parts = parsed.path.strip("/").split("/")
            if parts and parts[0] in SUPPORTED_LANGS:
                langs[parts[0]] = loc

        lastmod_tag = url_tag.find("lastmod")
        entry: dict = {"loc": loc, "langs": langs}
        if lastmod_tag:
            entry["lastmod"] = lastmod_tag.get_text(strip=True)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# URL map builder
# ---------------------------------------------------------------------------
def build_url_map(client: httpx.Client) -> tuple[dict[str, dict[str, str]], int]:
    """
    Crawl csasrl.it sitemaps and return:
      (url_map, filtered_count)

    url_map: { canonical_url: { "it": url, "en": url, "fr": url, "es": url } }
    filtered_count: number of URLs removed by BLOCKED_URL_PATTERNS
    """
    url_map: dict[str, dict[str, str]] = {}

    # 1. Fetch the sitemap index
    print(f"[scraper] Fetching sitemap index: {SITEMAP_URL}")
    index_xml = fetch(SITEMAP_URL, client)
    if not index_xml:
        print("[scraper] Could not fetch sitemap index.")
        return url_map, 0

    soup_test = BeautifulSoup(index_xml, "xml")
    is_index = bool(soup_test.find("sitemapindex"))

    if is_index:
        child_urls = parse_sitemap_index(index_xml, BASE_URL)
        print(f"[scraper] Found {len(child_urls)} child sitemaps.")
    else:
        # The root sitemap is itself a urlset
        child_urls = [SITEMAP_URL]
        print("[scraper] Root sitemap is a urlset (no index). Parsing directly.")

    # 2. Parse each child sitemap
    all_entries: list[dict] = []
    for sitemap_url in child_urls:
        print(f"[scraper]   Parsing: {sitemap_url}")
        xml = fetch(sitemap_url, client)
        if xml:
            entries = parse_url_set(xml)
            all_entries.extend(entries)
            print(f"    → {len(entries)} URLs")

    print(f"[scraper] Total URLs found (before filter): {len(all_entries)}")

    # 3. Build the map keyed on canonical URL, skipping blocked patterns
    filtered_count = 0
    for entry in all_entries:
        loc = entry["loc"]
        if _is_blocked(loc):
            print(f"  [filter] Skipping blocked URL: {loc}")
            filtered_count += 1
            continue
        langs = entry["langs"]
        url_map[loc] = langs

    print(f"[scraper] Filtered out {filtered_count} blocked URLs.")
    print(f"[scraper] Clean URLs to ingest: {len(url_map)}")

    return url_map, filtered_count


# ---------------------------------------------------------------------------
# Pinecone helpers (reused from pdf_ingest pattern)
# ---------------------------------------------------------------------------
def get_or_create_index(pc: Pinecone):
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
        # Extra warm-up: serverless indexes need ~10s after ready before accepting writes
        print("[pinecone] Index ready — waiting 15s for warm-up …")
        time.sleep(15)
    return pc.Index(PINECONE_INDEX_NAME)


def embed_texts(oai: OpenAI, texts: list[str]) -> list[list[float]]:
    response = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def upsert_url_vectors(url_map: dict, oai: OpenAI, index) -> int:
    """
    Embed each URL's canonical path as text and upsert with language URL metadata.
    This lets retrieval find relevant URLs when the user asks about a product/page.
    Blocked URLs are skipped as a safety net even if they slipped through build_url_map.
    """
    records = []
    for canonical, langs in url_map.items():
        # Safety-net: skip any blocked URL that might have slipped through
        if _is_blocked(canonical):
            continue
        # Use the path as the text to embed (descriptive enough for semantic search)
        parsed = urlparse(canonical)
        slug = parsed.path.strip("/").replace("/", " ").replace("-", " ")
        if not slug:
            continue
        vec_id = f"url__{canonical.replace('/', '_').replace(':', '_')}"[:512]
        records.append(
            {
                "id": vec_id,
                "text": slug,
                "metadata": {
                    "source_file": "web_scraper",
                    "type": "url_mapping",
                    "canonical_url": canonical,
                    "url_it": langs.get("it", ""),
                    "url_en": langs.get("en", ""),
                    "url_fr": langs.get("fr", ""),
                    "url_es": langs.get("es", ""),
                    "text": slug,
                },
            }
        )

    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        embeddings = embed_texts(oai, texts)
        vectors = [
            {"id": r["id"], "values": emb, "metadata": r["metadata"]}
            for r, emb in zip(batch, embeddings)
        ]
        # Retry upsert up to 3 times on timeout / transient errors
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
        print(f"  [pinecone] upserted {total}/{len(records)} URL vectors …")

    return total


# ---------------------------------------------------------------------------
# Delete stale URL vectors from Pinecone default namespace
# ---------------------------------------------------------------------------
def delete_url_vectors(index) -> int:
    """
    Delete all existing URL vectors from the default namespace.
    URL vectors are identified by their id prefix 'url__'.
    Uses paginated list() to collect all matching IDs, then deletes in batches.
    Returns count of deleted vectors.
    """
    print("[pinecone] Scanning default namespace for stale URL vectors …")
    ids_to_delete: list[str] = []

    try:
        # Pinecone's index.list() returns a paginated iterator.
        # Each page is a ListResponse with a .vectors attribute (list of ListItem with .id).
        for page in index.list(prefix="url__", limit=100):
            # page may be a ListResponse object or (in older SDKs) a plain list of strings
            if hasattr(page, "vectors"):
                for item in page.vectors:
                    ids_to_delete.append(item.id if hasattr(item, "id") else str(item))
            elif isinstance(page, list):
                for item in page:
                    ids_to_delete.append(item.id if hasattr(item, "id") else str(item))
            elif isinstance(page, str):
                ids_to_delete.append(page)
            else:
                # fallback: try treating page as a single item
                ids_to_delete.append(page.id if hasattr(page, "id") else str(page))
    except Exception as exc:
        print(f"[warn] list() not supported or failed ({exc}). Attempting delete_all by filter …")
        # Fallback: delete by metadata filter (requires paid Pinecone tier or different SDK)
        try:
            index.delete(filter={"type": {"$eq": "url_mapping"}})
            print("[pinecone] Deleted URL vectors via metadata filter.")
            return -1  # unknown count
        except Exception as exc2:
            print(f"[warn] Filter-based delete also failed ({exc2}). Skipping delete step.")
            return 0

    if not ids_to_delete:
        print("[pinecone] No existing URL vectors found to delete.")
        return 0

    print(f"[pinecone] Found {len(ids_to_delete)} URL vector IDs to delete.")
    DELETE_BATCH = 100
    deleted = 0
    for i in range(0, len(ids_to_delete), DELETE_BATCH):
        batch = ids_to_delete[i : i + DELETE_BATCH]
        index.delete(ids=batch)
        deleted += len(batch)
        print(f"  [pinecone] deleted {deleted}/{len(ids_to_delete)} …")

    print(f"[pinecone] Deletion complete: {deleted} vectors removed.")
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    # Step 1: delete stale URL vectors from default namespace
    deleted = delete_url_vectors(index)

    # Step 2: scrape fresh URLs (blocked patterns already filtered)
    with httpx.Client() as client:
        url_map, filtered_count = build_url_map(client)

    if not url_map:
        print("[scraper] No URLs scraped. Check network / sitemap structure.")
        return

    # Step 3: save url_map.json
    OUTPUT_PATH.write_text(json.dumps(url_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scraper] Saved url_map.json ({len(url_map)} entries) → {OUTPUT_PATH}")

    # Step 4: upsert fresh URL vectors
    n = upsert_url_vectors(url_map, oai, index)
    print(f"\n[scraper] === SUMMARY ===")
    print(f"  Deleted stale vectors : {deleted}")
    print(f"  URLs scraped (raw)    : {len(url_map) + filtered_count}")
    print(f"  Filtered out (blocked): {filtered_count}")
    print(f"  Clean URLs in map     : {len(url_map)}")
    print(f"  Vectors upserted      : {n}")
    print(f"  url_map.json saved to : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
