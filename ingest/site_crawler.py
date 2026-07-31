"""
ingest/site_crawler.py
----------------------
Crawls csasrl.it and indexes the content of pages that are actually reachable
by navigating the site.

Why this replaces the sitemap-only approach in ingest/web_scraper.py
--------------------------------------------------------------------
1. Reachability. The sitemap lists 900+ URLs including pages nothing links to —
   /prova-menu-prodotti/, /account/, /log-in/, and 43 /tag-prodotto/ archives.
   Those are not part of the site a visitor can navigate, so the chatbot must
   not cite them. Reachability is decided by a breadth-first walk of the link
   graph from the four language homepages; the sitemap is used only for its
   hreflang data.

2. Content. web_scraper.py embedded the URL *slug* as the vector text ("prodotti
   valvole xlc 310"), so the bot could produce a link but knew nothing about
   what the page said. This indexes the page body, which for a product page is
   the pressure ratings, materials, flange range and applications.

Each chunk carries url_it/url_en/url_fr/url_es, so retrieval can return the link
in the reader's language regardless of which language edition matched.

Run standalone:
    python -m ingest.site_crawler
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from ingest.pdf_ingest import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    BATCH_SIZE,
    _batch,
    chunk_text,
    embed_texts,
    get_or_create_index,
)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
URL_MAP_PATH = REPO_ROOT / "url_map.json"
STRUCTURE_PATH = REPO_ROOT / "api" / "site_structure.json"

BASE_URL = "https://csasrl.it"
SITEMAP_INDEX = f"{BASE_URL}/sitemap_index.xml"
SUPPORTED_LANGS = ("it", "en", "fr", "es")

# Language homepages: the Italian home links to the other three, but seeding all
# four keeps a broken language switcher from hiding an entire edition.
SEEDS = [f"{BASE_URL}/"] + [f"{BASE_URL}/{lang}/" for lang in ("en", "fr", "es")]

MAX_PAGES = 1500
MAX_DEPTH = 6
CONCURRENCY = 4          # small pool: this is a live customer site
REQUEST_DELAY = 0.25     # seconds between request starts
ID_PREFIX = "page__"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# URL exclusions
# ---------------------------------------------------------------------------
# E-commerce, account and admin paths. '/negozio/' is the Italian shop and was
# missing from the original blocklist, which only covered the English '/shop/'.
BLOCKED_PATTERNS: tuple[str, ...] = (
    "/shop/", "/negozio/", "/boutique/", "/tienda/",
    "/cart/", "/carrello/", "/panier/", "/carrito/",
    "/checkout/", "/cassa/", "/pagamento/", "/commande/",
    "/my-account/", "/mio-account/", "/account", "/mon-compte/", "/mi-cuenta/",
    "/log-in/", "/login/", "/logout", "/register", "/password-reset",
    "/edit-profile", "/members", "/user-", "/user/", "/utente/", "/usuario/",
    "/wp-admin/", "/wp-content/", "/wp-json/", "/wp-login",
    # Tag archives: thin listing pages that only enumerate products already
    # indexed in full. The site uses '/tag-product/', not WordPress's default
    # '/product-tag/', so matching only the default let 58 of them through.
    "/tag-prodotto/", "/tag-product/", "/product-tag/",
    "/etiqueta-producto/", "/etiquette-produit/",
    "/rimborso_reso/", "/add-to-cart",
)

# Paginated listings ('/categoria-prodotto/idrovalvole/page/2/') must be walked
# but not indexed. Their own content is a near-duplicate of page 1, yet each
# page links to a different slice of products — skipping them entirely dropped
# 32 product pages per language, because page 2 onwards is the only route to
# them once the tag archives are excluded.
PAGINATION_RE = re.compile(r"/page/\d+/?$")

# Non-HTML endpoints reached from content links.
SKIP_SUFFIXES = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".rar",
    ".dwg", ".dxf", ".stp", ".step", ".xls", ".xlsx", ".doc", ".docx", ".mp4",
    ".css", ".js", ".xml", ".ico",
)


def is_blocked(url: str) -> bool:
    """True when *url* must never be crawled or cited."""
    lower = url.lower()
    if any(pattern in lower for pattern in BLOCKED_PATTERNS):
        return True
    path = urlparse(lower).path
    return path.endswith(SKIP_SUFFIXES)


def is_indexable(url: str) -> bool:
    """
    True when a crawled page's own content belongs in the index.

    Pagination is crawled for the product links it carries but never indexed or
    cited: it is the same listing sliced differently.
    """
    return not PAGINATION_RE.search(urlparse(url).path)


def normalize(url: str) -> str:
    """
    Canonical form of a URL for de-duplication.

    Fragments and query strings are dropped (they address the same document) and
    a trailing slash is enforced so '/chi-siamo' and '/chi-siamo/' are one page.
    """
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path or "/"
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


def is_internal(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc.lower().endswith("csasrl.it")


def language_of(url: str) -> str:
    """Language edition a URL belongs to, from its leading path segment."""
    segments = urlparse(url).path.strip("/").split("/")
    if segments and segments[0] in ("en", "fr", "es"):
        return segments[0]
    return "it"


def section_of(url: str) -> str:
    """
    Top-level site section, used to describe the architecture.

    The language prefix is skipped so the English '/en/products/' and the
    Italian '/prodotti/' both report their own section name rather than 'en'.
    """
    segments = [s for s in urlparse(url).path.strip("/").split("/") if s]
    if segments and segments[0] in ("en", "fr", "es"):
        segments = segments[1:]
    return segments[0] if segments else "(home)"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
async def fetch(client: httpx.AsyncClient, url: str, retries: int = 3) -> str | None:
    """GET *url*, returning HTML text or None after exhausting retries."""
    for attempt in range(retries):
        try:
            response = await client.get(url, headers=HEADERS, timeout=25.0, follow_redirects=True)
            if response.status_code >= 400:
                return None
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "xml" not in content_type:
                return None
            return response.text
        except httpx.HTTPError:
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# hreflang map from the sitemap
# ---------------------------------------------------------------------------
async def load_hreflang_map(client: httpx.AsyncClient) -> dict[str, dict[str, str]]:
    """
    Return {url: {lang: url}} built from the sitemap's xhtml:link entries.

    The pages carry no hreflang tags in their markup, so the sitemap is the only
    source that ties the four language editions of a page together — the slugs
    are translated ('/prodotti/' vs '/en/products/'), so paths cannot be matched.
    """
    index_xml = await fetch(client, SITEMAP_INDEX)
    if not index_xml:
        print("[crawl] [warn] sitemap index unavailable — language links limited")
        return {}

    soup = BeautifulSoup(index_xml, "xml")
    child_urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

    mapping: dict[str, dict[str, str]] = {}
    for child in child_urls:
        xml = await fetch(client, child)
        if not xml:
            continue
        child_soup = BeautifulSoup(xml, "xml")
        for entry in child_soup.find_all("url"):
            loc_tag = entry.find("loc")
            if not loc_tag:
                continue
            loc = normalize(loc_tag.get_text(strip=True))
            langs: dict[str, str] = {}
            for link in entry.find_all("link"):
                code = (link.get("hreflang") or "").split("-")[0].lower()
                href = link.get("href") or ""
                if code in SUPPORTED_LANGS and href and not is_blocked(href):
                    langs[code] = normalize(href)
            if langs:
                mapping[loc] = langs
        print(f"[crawl]   sitemap {child.rsplit('/', 1)[-1]}: {len(mapping)} cumulative entries")

    return mapping


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
NOISE_PHRASES = ("Vai al contenuto", "Skip to content", "Aller au contenu", "Ir al contenido")


def extract_content(html: str) -> tuple[str, str, str, list[str]]:
    """
    Return (title, meta_description, body_text, internal_links) for a page.

    Chrome shared by every page — menus, header, footer, forms, scripts — is
    stripped so a chunk holds the page's own content rather than the navigation
    repeated 900 times across the index.
    """
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = (meta.get("content") or "").strip() if meta else ""

    links = [a["href"] for a in soup.find_all("a", href=True)]

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "form", "iframe", "svg", "button"]):
        tag.decompose()

    body = soup.body
    text = re.sub(r"\s+", " ", body.get_text(" ")).strip() if body else ""
    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, "").strip()

    return title, description, text, links


# ---------------------------------------------------------------------------
# Breadth-first crawl
# ---------------------------------------------------------------------------
async def crawl(client: httpx.AsyncClient) -> dict[str, dict]:
    """
    Walk the link graph from the language homepages.

    Returns {url: {title, description, text, depth, lang, section}} for every
    reachable page. A page absent from this map is unreachable by navigation and
    is deliberately never indexed.
    """
    pages: dict[str, dict] = {}
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for seed in SEEDS:
        url = normalize(seed)
        seen.add(url)
        queue.append((url, 0))

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def visit(url: str, depth: int) -> list[str]:
        async with semaphore:
            await asyncio.sleep(REQUEST_DELAY)
            html = await fetch(client, url)
        if not html:
            return []

        title, description, text, links = extract_content(html)
        pages[url] = {
            "title": title,
            "description": description,
            "text": text,
            "depth": depth,
            "lang": language_of(url),
            "section": section_of(url),
        }

        found: list[str] = []
        for href in links:
            absolute = normalize(urljoin(url, href))
            if not is_internal(absolute) or is_blocked(absolute):
                continue
            found.append(absolute)
        return found

    while queue and len(pages) < MAX_PAGES:
        # Drain one depth level at a time so the pool stays busy.
        batch: list[tuple[str, int]] = []
        while queue and len(batch) < CONCURRENCY * 4:
            batch.append(queue.popleft())

        results = await asyncio.gather(*(visit(u, d) for u, d in batch))

        for (_, depth), discovered in zip(batch, results):
            if depth >= MAX_DEPTH:
                continue
            for link in discovered:
                if link not in seen:
                    seen.add(link)
                    queue.append((link, depth + 1))

        print(f"[crawl]   visited={len(pages)} queued={len(queue)}")

    return pages


# ---------------------------------------------------------------------------
# Pages behind the customer login
# ---------------------------------------------------------------------------
# csasrl.it keeps its sizing programs (CSA CVS for the XLC series, the air-valve
# sizing program, the VRCD/GEMINA/ATHENA/ITALICA calculators) in a members area.
# To an anonymous crawler the hub page lists no links and each program renders a
# login form, so the link-graph walk cannot see them — yet they are real pages a
# registered customer uses, and the chatbot has to be able to point at them.
#
# They are therefore taken from the sitemap, verified live, and indexed as
# pointers: what the tool is, in which language, and that it needs a login. No
# content is invented — the gated body is never read.
GATED_BODY_MAX_CHARS = 700

# Slugs that mark a page as unfinished or a leftover experiment. A page nothing
# links to is usually still legitimate — the public Italica/Athena/Gemina
# calculators and two refund policies are all unlinked but real — so only these
# markers, not unlinkedness on its own, keep a page out of the index.
DRAFT_SLUG_MARKERS = ("bozza", "draft", "-test", "test-", "prova", "copia", "-copy", "-old")


def looks_like_draft(url: str) -> bool:
    """True when the slug marks the page as a draft or leftover experiment."""
    path = urlparse(url).path.lower()
    return any(marker in path for marker in DRAFT_SLUG_MARKERS)

# Slug fragments that should be printed as acronyms rather than capitalised.
_ACRONYMS = {
    "csa", "cvs", "rvs", "prs", "lvs", "avs", "xlc", "vrcd", "rfp", "dn",
    "3f", "ac", "cp", "iso", "pn",
}

_GATED_NOTE = {
    "it": ("Contenuto riservato del sito csasrl.it: per usarlo serve registrarsi "
           "e accedere all'area clienti."),
    "en": ("Members-only content on csasrl.it: registration and login are required "
           "to use it."),
    "fr": ("Contenu réservé du site csasrl.it : l'inscription et la connexion à "
           "l'espace client sont nécessaires pour l'utiliser."),
    "es": ("Contenido reservado del sitio csasrl.it: es necesario registrarse e "
           "iniciar sesión en el área de clientes para usarlo."),
}


def _product_families() -> set[str]:
    """Family names from the model registry, used to capitalise slug words."""
    try:
        registry = json.loads((REPO_ROOT / "api" / "model_registry.json").read_text("utf-8"))
        return set(registry.get("families", {}))
    except (OSError, json.JSONDecodeError):
        return set()


_FAMILIES = _product_families()


def title_from_slug(url: str) -> str:
    """
    Build a readable title from a URL slug.

    The gated Italian pages return "Login - CSA SRL …" as their title, so the
    slug is the only description available for them.
    """
    slug = urlparse(url).path.strip("/").split("/")[-1]
    words = [w for w in slug.replace("_", "-").split("-") if w]
    out = []
    for word in words:
        lower = word.lower()
        if lower in _ACRONYMS:
            out.append(word.upper())
        elif lower in _FAMILIES:
            out.append(word.capitalize())
        else:
            out.append(word)
    text = " ".join(out)
    return text[:1].upper() + text[1:] if text else ""


async def discover_unlinked_pages(
    client: httpx.AsyncClient,
    hreflang: dict[str, dict[str, str]],
    crawled: set[str],
) -> dict[str, dict]:
    """
    Recover sitemap pages the link walk could not reach.

    Every sitemap URL the crawl missed is fetched once and sorted into three
    kinds:

    * **Members area** — the body is essentially empty because the page renders
      a login form. These are the sizing programs (CSA CVS for the XLC series,
      the air-valve program, the VRCD/Gemina/Athena/Italica calculators). They
      are indexed as pointers: what the tool is and that it needs a login. The
      gated body is never read.
    * **Draft** — the slug marks it as unfinished, like '/en/home-eng-bozza/'.
      Dropped.
    * **Unlinked but real** — renders proper content and simply has no inbound
      link, like the public Italica and Athena calculators and two refund
      policies. Indexed normally: the page works, nothing about it is a mistake.
    """
    candidates = [
        url for url in hreflang
        if url not in crawled and not is_blocked(url) and is_indexable(url)
    ]
    print(f"[crawl] Checking {len(candidates)} sitemap URLs the walk did not reach …")

    found: dict[str, dict] = {}
    gated_count = 0
    drafts: list[str] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def check(url: str) -> None:
        nonlocal gated_count
        if looks_like_draft(url):
            drafts.append(url)
            return

        async with semaphore:
            await asyncio.sleep(REQUEST_DELAY)
            html = await fetch(client, url)
        if not html:
            return

        title, description, text, _ = extract_content(html)
        lang = language_of(url)
        section = section_of(url)
        gated = len(text) <= GATED_BODY_MAX_CHARS

        if gated:
            # The Italian editions report the login form's title; the English
            # and French ones keep the real page title even while gated.
            if not title or title.lower().startswith("login"):
                title = title_from_slug(url)
            else:
                title = title.split(" - CSA SRL")[0].strip()
            body = f"{title}. {_GATED_NOTE.get(lang, _GATED_NOTE['en'])} ({section})"
            gated_count += 1
        else:
            body = text

        found[url] = {
            "title": title,
            "description": description,
            "text": body,
            "depth": -1,
            "lang": lang,
            "section": section,
            "gated": gated,
        }

    await asyncio.gather(*(check(u) for u in candidates))

    print(f"[crawl] Recovered {len(found)} unlinked pages "
          f"({gated_count} in the members area, {len(found) - gated_count} public).")
    print(f"[crawl] Drafts skipped: {len(drafts)}")
    for url in sorted(drafts):
        print(f"[crawl]   draft: {url}")
    return found


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
# Vectors written by the superseded ingest/web_scraper.py. They hold a URL slug
# as their only text and cover every sitemap entry, including the orphan pages
# this crawl exists to exclude — leaving them in place would let the bot keep
# citing /prova-menu-prodotti/ and the tag archives.
LEGACY_PREFIX = "url__"


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
        print(f"[crawl] [warn] could not list '{prefix}' vectors ({exc})")
    return ids


def delete_existing(index) -> int:
    """Remove this script's previous output and the legacy URL-slug vectors."""
    removed = 0
    for prefix, label in ((ID_PREFIX, "page-content"), (LEGACY_PREFIX, "legacy URL-slug")):
        ids = _ids_with_prefix(index, prefix)
        for i in range(0, len(ids), 100):
            index.delete(ids=ids[i : i + 100])
        if ids:
            print(f"[crawl] Removed {len(ids)} {label} vectors.")
        removed += len(ids)
    return removed


def build_records(pages: dict[str, dict], hreflang: dict[str, dict[str, str]]) -> list[dict]:
    """Turn crawled pages into Pinecone records, one per content chunk."""
    records: list[dict] = []

    for url, page in sorted(pages.items()):
        langs = hreflang.get(url, {})
        # Fall back to self-linking so a page missing from the sitemap still
        # yields a usable link in at least its own language.
        if not langs:
            langs = {page["lang"]: url}
        # Only cite reachable translations: an unreachable variant is exactly
        # the kind of orphan page this crawl exists to avoid.
        langs = {code: target for code, target in langs.items() if target in pages}
        if not langs:
            langs = {page["lang"]: url}

        header_parts = [part for part in (page["title"], page["description"]) if part]
        header = " — ".join(header_parts)
        full_text = f"{header}\n{page['text']}" if header else page["text"]
        if len(full_text.strip()) < 80:
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", urlparse(url).path.strip("/").lower()) or "home"
        for chunk_idx, chunk in enumerate(chunk_text(full_text)):
            # Repeat the title in every chunk: later chunks otherwise lose all
            # trace of which product they describe.
            body = chunk if chunk_idx == 0 or not page["title"] else f"{page['title']}\n{chunk}"
            vector_id = f"{ID_PREFIX}{slug}__c{chunk_idx}"[:500]
            records.append(
                {
                    "id": vector_id,
                    "text": body,
                    "metadata": {
                        "source_file": "csasrl.it",
                        "type": "page_content",
                        "chunk_id": vector_id,
                        "chunk_index": chunk_idx,
                        "text": body,
                        "canonical_url": langs.get("it", url),
                        "url_it": langs.get("it", ""),
                        "url_en": langs.get("en", ""),
                        "url_fr": langs.get("fr", ""),
                        "url_es": langs.get("es", ""),
                        "page_title": page["title"][:300],
                        "lang": page["lang"],
                        "section": page["section"],
                    },
                }
            )

    return records


def upsert(records: list[dict], oai: OpenAI, index) -> int:
    total = 0
    for batch in _batch(records, BATCH_SIZE):
        embeddings = embed_texts(oai, [r["text"] for r in batch])
        vectors = [
            {"id": r["id"], "values": emb, "metadata": r["metadata"]}
            for r, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors, timeout=120)
        total += len(vectors)
        print(f"  [pinecone] upserted {total}/{len(records)} …")
    return total


def write_site_structure(pages: dict[str, dict], hreflang: dict[str, dict[str, str]]) -> dict:
    """
    Save the reachable-page inventory grouped by section and language.

    api/site_structure.json ships with the app so the chatbot can answer "what
    is on the site" from the pages that genuinely exist, and so a later run can
    be diffed against this one.
    """
    sections: dict[str, dict[str, list[str]]] = {}
    for url, page in sorted(pages.items()):
        section = sections.setdefault(page["section"], {})
        section.setdefault(page["lang"], []).append(url)

    structure = {
        "_comment": (
            "Reachable pages on csasrl.it, discovered by breadth-first crawl of "
            "the link graph. Generated by ingest/site_crawler.py."
        ),
        "total_pages": len(pages),
        "sections": {
            name: {lang: len(urls) for lang, urls in langs.items()}
            for name, langs in sorted(sections.items())
        },
        "pages_by_section": sections,
    }
    STRUCTURE_PATH.write_text(
        json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return structure


async def main() -> None:
    async with httpx.AsyncClient() as client:
        print("[crawl] Loading hreflang map from sitemap …")
        hreflang = await load_hreflang_map(client)
        print(f"[crawl] hreflang entries: {len(hreflang)}")

        print("[crawl] Walking the link graph from the language homepages …")
        pages = await crawl(client)

        unlinked = await discover_unlinked_pages(client, hreflang, set(pages))

    # Pagination was crawled only to reach the products it links to; from here
    # on it is not a page the chatbot may describe or cite.
    crawled = len(pages)
    pages = {url: page for url, page in pages.items() if is_indexable(url)}
    print(f"[crawl] Reachable pages: {len(pages)} "
          f"({crawled - len(pages)} pagination pages walked but not indexed)")

    pages.update(unlinked)
    print(f"[crawl] Total indexable pages: {len(pages)} "
          f"({len(unlinked)} recovered from the sitemap)")

    reachable_map = {
        url: {code: target for code, target in hreflang.get(url, {}).items() if target in pages}
        for url in sorted(pages)
    }
    URL_MAP_PATH.write_text(
        json.dumps(reachable_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[crawl] url_map.json rewritten with {len(reachable_map)} reachable URLs.")

    structure = write_site_structure(pages, hreflang)
    print(f"[crawl] site_structure.json: {structure['total_pages']} pages across "
          f"{len(structure['sections'])} sections.")

    records = build_records(pages, hreflang)
    print(f"[crawl] {len(records)} content chunks to index.")

    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)
    delete_existing(index)
    total = upsert(records, oai, index)

    print(f"\n[crawl] Done. {total} page-content vectors upserted.")


if __name__ == "__main__":
    asyncio.run(main())
