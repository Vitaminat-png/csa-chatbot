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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CSAChatbotScraper/1.0; "
        "+https://csasrl.it)"
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
def build_url_map(client: httpx.Client) -> dict[str, dict[str, str]]:
    """
    Crawl csasrl.it sitemaps and return:
      { canonical_url: { "it": url, "en": url, "fr": url, "es": url } }
    """
    url_map: dict[str, dict[str, str]] = {}

    # 1. Fetch the sitemap index
    print(f"[scraper] Fetching sitemap index: {SITEMAP_URL}")
    index_xml = fetch(SITEMAP_URL, client)
    if not index_xml:
        print("[scraper] Could not fetch sitemap index.")
        return url_map

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

    print(f"[scraper] Total URLs found: {len(all_entries)}")

    # 3. Build the map keyed on canonical URL
    for entry in all_entries:
        loc = entry["loc"]
        langs = entry["langs"]
        url_map[loc] = langs

    return url_map


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
    """
    records = []
    for canonical, langs in url_map.items():
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    with httpx.Client() as client:
        url_map = build_url_map(client)

    if not url_map:
        print("[scraper] No URLs scraped. Check network / sitemap structure.")
        return

    # Save JSON
    OUTPUT_PATH.write_text(json.dumps(url_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scraper] Saved url_map.json ({len(url_map)} entries) → {OUTPUT_PATH}")

    # Upsert URL vectors
    n = upsert_url_vectors(url_map, oai, index)
    print(f"[scraper] Upserted {n} URL vectors to Pinecone.")


if __name__ == "__main__":
    main()
