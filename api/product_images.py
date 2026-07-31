"""
api/product_images.py
---------------------
Mapping from CSA product families / valve models to product image URLs.

HOW TO UPDATE:
  Replace the placeholder paths under each family key with the real image URL
  from csasrl.it (e.g. from the WordPress media library).
  You can also add valve-model-level overrides in VALVE_MODEL_IMAGES.

URL strategy:
  - Primary: /static/products/<slug>.jpg  (self-hosted, served by FastAPI)
  - Fallback: full https://www.csasrl.it/... URL (hotlink from CSA website)
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Family-level images
# Key = product_family string as stored in Pinecone metadata (lowercase-normalised
#       during lookup, so case doesn't matter at call time).
# Value = dict with:
#   url          – image src; use absolute URL or /static/products/<file>
#   alt          – descriptive alt text
#   product_name – human-readable product name shown in the widget
# ---------------------------------------------------------------------------
PRODUCT_IMAGES: dict[str, dict] = {
    # Gate valves
    "xlc": {
        "url": "/static/products/xlc.jpg",
        "alt": "Valvola a saracinesca XLC",
        "product_name": "Valvola a saracinesca XLC",
    },
    # Butterfly valves
    "argo": {
        "url": "/static/products/argo.jpg",
        "alt": "Valvola a farfalla ARGO",
        "product_name": "Valvola a farfalla ARGO",
    },
    # Ball valves
    "italica 353": {
        "url": "/static/products/italica353.jpg",
        "alt": "Valvola a sfera ITALICA 353",
        "product_name": "Valvola a sfera ITALICA 353",
    },
    "italica": {
        "url": "/static/products/italica353.jpg",
        "alt": "Valvola a sfera ITALICA",
        "product_name": "Valvola a sfera ITALICA",
    },
    # Check valves
    "protector": {
        "url": "/static/products/protector.jpg",
        "alt": "Valvola di ritegno PROTECTOR",
        "product_name": "Valvola di ritegno PROTECTOR",
    },
    # Couplings / joints
    "dedalo": {
        "url": "/static/products/dedalo.jpg",
        "alt": "Giunto DEDALO",
        "product_name": "Giunto DEDALO",
    },
    # Fire hydrants
    "vortice": {
        "url": "/static/products/vortice.jpg",
        "alt": "Idrante VORTICE",
        "product_name": "Idrante VORTICE",
    },
    # Additional common families
    "orbis": {
        "url": "/static/products/orbis.jpg",
        "alt": "Valvola ORBIS",
        "product_name": "Valvola ORBIS",
    },
    "isis": {
        "url": "/static/products/isis.jpg",
        "alt": "Valvola ISIS",
        "product_name": "Valvola ISIS",
    },
}

# Optional valve-model-level overrides (higher priority than family)
# Key = valve_model string (lowercase-normalised)
VALVE_MODEL_IMAGES: dict[str, dict] = {
    # Example: "xlc 100": {"url": "...", "alt": "...", "product_name": "..."},
}

MAX_IMAGES_PER_RESPONSE = 2

# ---------------------------------------------------------------------------
# Dimension drawings
# ---------------------------------------------------------------------------
# (source_file, page) -> PNG of that datasheet page, rendered by
# ingest/render_dimension_pages.py. The page holding the dimensions table also
# holds the quoted drawing that gives the letters their meaning: an answer
# listing "A = 230 mm, B = 82,5 mm" without it is a list of blind numbers.
# The map is keyed by the page the chunk came from, so the drawing appears
# exactly when the table it explains was actually used to answer.
_DRAWINGS_PATH = Path(__file__).resolve().parent / "dimension_drawings.json"

try:
    _DIMENSION_DRAWINGS: dict[str, dict[str, str]] = json.loads(
        _DRAWINGS_PATH.read_text(encoding="utf-8")
    )
except (FileNotFoundError, json.JSONDecodeError):
    _DIMENSION_DRAWINGS = {}


def _drawing_name(source_file: str, url: str) -> str:
    """Human name for the drawing: 'Dimensioni FOX 3F', 'Dimensioni serie XLC 400'."""
    stem = url.rsplit("/", 1)[-1].removesuffix(".png")
    if stem.startswith("xlc_serie_"):
        return f"Dimensioni serie XLC {stem.rsplit('_', 1)[-1]}"
    model = source_file.removesuffix(".pdf").replace("_", " ")
    return f"Dimensioni {model}"


def get_dimension_drawings(sources: list, named_files: tuple[str, ...] = ()) -> list[dict]:
    """
    Drawings for the datasheet pages the answer actually drew on.

    *sources* are api.models.Source objects (or anything with .source_file,
    .page and .is_exact_model). *named_files* are the datasheets the question
    itself names (find_model_sources on the search query): when present, only
    their sources may contribute a drawing — retrieval routinely pads the
    context with other products' dimension pages, and a question about the
    ATHENA was answered with the ATHENA drawing plus a stray XLC one.

    The is_exact_model flag alone was not enough to know the topic: it is set
    from the canonical registry, which requires a two-token code, so a
    single-word model — "che misure ha l'athena" — never sets it, the filter
    fell back to the leading source (a catalogue page), and no drawing was
    attached at all while ATHENA.pdf p3 sat two positions down.
    """
    named = set(named_files or ())
    if named:
        pool = [s for s in sources if getattr(s, "source_file", "") in named]
        # Within the named files, a source marked exact (the named model's own
        # datasheet, or the series table the question asks about) narrows it
        # further: for "XLC 400 DN 300" both series tables can be in context,
        # and only the 400's is marked — its drawing is the one to show.
        exact = [s for s in pool if getattr(s, "is_exact_model", False)]
        pool = exact or pool
    else:
        # No model recognised in the question: exact-flagged sources may still
        # carry a drawing (a series named without the registry), but there is
        # no falling back to "whatever source leads". That guess attached the
        # XLC 300 drawing to a REFUSAL about a product the bot claimed not to
        # know — an unrecognised question gets no drawing at all.
        pool = [s for s in sources if getattr(s, "is_exact_model", False)]

    results: list[dict] = []
    seen: set[str] = set()
    for src in pool:
        pages = _DIMENSION_DRAWINGS.get(getattr(src, "source_file", "") or "")
        if not pages:
            continue
        url = pages.get(str(getattr(src, "page", "") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "url": url,
            "alt": f"Sagoma quotata — {_drawing_name(src.source_file, url)}",
            "product_name": _drawing_name(src.source_file, url),
        })
        if len(results) >= MAX_IMAGES_PER_RESPONSE:
            break
    return results


def get_images_for_families(
    product_families: list[str],
    valve_models: list[str] | None = None,
) -> list[dict]:
    """
    Given a list of product_family strings (and optionally valve_model strings)
    extracted from retrieved chunks, return up to MAX_IMAGES_PER_RESPONSE image
    descriptors.

    Returns a list of dicts: [{url, alt, product_name}, ...]
    """
    seen_urls: set[str] = set()
    results: list[dict] = []

    def _add(entry: dict) -> None:
        if entry["url"] in seen_urls or len(results) >= MAX_IMAGES_PER_RESPONSE:
            return
        # Self-hosted entries whose file was never provided are skipped: the
        # placeholder map shipped before any photo existed, and every answer
        # about those families carried a dead /static URL the widget had to
        # 404 on and hide. Drop a real photo into static/products/ and the
        # entry starts working, with no code change.
        if entry["url"].startswith("/static/"):
            file_path = Path(__file__).resolve().parent.parent / entry["url"].lstrip("/")
            if not file_path.exists():
                return
        seen_urls.add(entry["url"])
        results.append(entry)

    # Check valve-model overrides first
    for vm in (valve_models or []):
        key = vm.strip().lower()
        if key in VALVE_MODEL_IMAGES:
            _add(VALVE_MODEL_IMAGES[key])

    # Then family-level images
    for fam in product_families:
        key = fam.strip().lower()
        if key in PRODUCT_IMAGES:
            _add(PRODUCT_IMAGES[key])

    return results
