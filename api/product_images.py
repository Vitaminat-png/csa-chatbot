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
        if entry["url"] not in seen_urls and len(results) < MAX_IMAGES_PER_RESPONSE:
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
