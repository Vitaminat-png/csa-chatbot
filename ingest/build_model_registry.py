"""
ingest/build_model_registry.py
------------------------------
Builds api/model_registry.json: a map from valve model codes to the datasheet
filenames that document them.

Why this exists
---------------
Cosine similarity on embeddings cannot tell "ITALICA 353" from "ITALICA 310" —
the two strings are near-identical in vector space, so an Italian question about
the 353 reliably retrieved 310 content and the bot answered "no information"
while ITALICA_353.pdf sat in the index. Quoting the wrong pressure rating for an
industrial valve is a safety problem, so model codes get exact lexical matching
instead of relying on semantic search alone.

The registry ships inside api/ on purpose: docs/ is excluded from the Docker
image (.dockerignore) and data/*.json is gitignored, so neither is readable in
production.

Run standalone:
    python -m ingest.build_model_registry
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
# Schede tolte deliberatamente dall'indice: il prodotto resta visibile dalle
# pagine del sito, ma la sua scheda tecnica non deve essere una fonte. I loro
# codici vanno comunque registrati, altrimenti una domanda che nomina il
# modello escluso ricade sulla chiave piu' corta e riceve la scheda di un
# ALTRO prodotto: tolto APOLLO_RPC_SMART.pdf, "Apollo RPC SMART" risolveva
# APOLLO_RPC.pdf — con l'etichetta "questa e' la scheda del modello chiesto".
EXCLUDED_DIR = DOCS_DIR / "_esclusi"
XLC_DIR = REPO_ROOT / "xlc engeniering"
OUTPUT_PATH = REPO_ROOT / "api" / "model_registry.json"

# Filenames that are whole catalogues rather than a single model's datasheet.
# They cover every family, so attaching them to one model code would poison it.
CATALOGUE_MARKERS = ("catalogo", "catalogue", "catalog")

# Tokens that appear in filenames but are variant suffixes, not part of the
# model identity users type (e.g. FOX_3F_AS_HP -> family FOX, variant 3F AS HP).
_FAMILY_STOPWORDS = {"v2", "eng", "ita", "ital", "fra", "spa", "en", "it", "fr", "es"}


def _tokenize_stem(stem: str) -> list[str]:
    """
    Split a filename stem into lowercase model tokens.

    'XLC_310_DC'          -> ['xlc', '310', 'dc']
    'XLC engineering ENG' -> ['xlc', 'engineering', 'eng']
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", stem)
    # Separate letter/digit runs so 'ATHENA114' tokenises like 'ATHENA_114'
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    return [t.lower() for t in cleaned.split() if t]


def _is_catalogue(stem: str) -> bool:
    lower = stem.lower()
    return any(marker in lower for marker in CATALOGUE_MARKERS)


def file_digest(path: Path) -> str:
    """MD5 of a file's bytes, used to drop byte-identical duplicates."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dedupe_by_content(paths: list[Path]) -> list[Path]:
    """
    Keep one path per distinct file content.

    The XLC folder ships each language twice ('XLC engineering ENG v2.pdf' and
    'XLC-engineering-ENG.pdf' are the same bytes). Ingesting both would double
    the vectors and let a duplicate occupy a second context slot.
    """
    seen: dict[str, Path] = {}
    for path in sorted(paths):
        digest = file_digest(path)
        if digest not in seen:
            seen[digest] = path
    return sorted(seen.values())


# Applications a valve is documented for. Users ask "what do you recommend for
# irrigation?" far more often than they ask about a model code, and no amount of
# chunk similarity can enumerate the 32 datasheets that name irrigation — the
# bot listed two of them and then reported it had nothing further.
#
# Keys are English because the query side matches against the English rendering
# of the question, but the *patterns* must cover Italian, French and Spanish
# too: the XLC engineering set ships in four languages, and English-only
# patterns read the same document four different ways — the English edition came
# out as "industrial, water supply", the Spanish one as "drinking water" (its
# "agua potable" happening to contain the English "potable"), and the Italian
# and French editions as nothing at all.
APPLICATION_PATTERNS: dict[str, str] = {
    "irrigation": r"irrigat|irrigaz|riego|regad",
    "sewage": (
        r"sewage|waste ?water|sewer|fognat|fogna\b|acque reflue"
        r"|eaux usées|égout|alcantarill|aguas residuales"
    ),
    "drinking water": r"drinking water|potable|potabil|agua potable|eau potable",
    "sea water": (
        r"sea ?water|desalinat|marine|acqua di mare|acqua marina"
        r"|eau de mer|agua de mar|dissalaz"
    ),
    "fire fighting": (
        r"fire[- ]?fighting|hydrant|fire protection|idrant|antincendio"
        r"|incendie|incendio|bouche d'incendie|hidrante"
    ),
    "pumping station": (
        r"pumping station|pump station|downstream of pumps|stazion[ei] di pompaggio"
        r"|a valle di pompe|station de pompage|estaci[oó]n de bombeo"
    ),
    "industrial": (
        r"industrial (?:plant|application|use)|impiant\w* industrial\w*"
        r"|uso industriale|sites? industriels?|plantas? industriales?"
        r"|refiner|petrochemical|process plant|raffiner|petrolchimic"
    ),
    "water supply network": (
        r"water supply|distribution network|water main|acquedott|adduzione"
        r"|reti? di distribuzione|réseaux? de distribution|adduction"
        r"|redes? de distribuci[oó]n|abastecimiento"
    ),
    "mining": r"\bmining\b|\bmine[sd]?\b|minerar|miniera|minero|exploitation minière",
    "deep well": r"deep well|pozzo profondo|puits profond|pozo profundo|borehole",
}


def _language_stripped_stem(path: Path) -> str:
    """
    Identity of a document across its language editions.

    'XLC engineering ENG v2' and 'XLC engineering ITAL v2' are the same document,
    so they must report the same applications.
    """
    tokens = [t for t in _tokenize_stem(path.stem) if t not in _FAMILY_STOPWORDS]
    return " ".join(tokens)


def unify_translations(applications: dict[str, set[str]], paths: list[Path]) -> None:
    """
    Give every language edition of a document the union of their applications.

    Extraction is regex-based per language, and no realistic set of patterns
    catches every wording in four of them: the same XLC engineering page reads
    as "Industrial plants", "Sites industriels" and "Plantas industriales", and
    a gap in one language silently dropped that application for that edition.
    Since these files are translations of one document, the union is the honest
    answer for all of them.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[_language_stripped_stem(path)].append(path.name)

    for names in groups.values():
        if len(names) < 2:
            continue
        shared = {
            app for app, files in applications.items()
            if any(n in files for n in names)
        }
        for app in shared:
            applications[app].update(names)


def extract_applications(text: str) -> list[str]:
    """Return the applications named in a datasheet's text."""
    lowered = text.lower()
    return [
        name for name, pattern in APPLICATION_PATTERNS.items()
        if re.search(pattern, lowered)
    ]


def read_pdf_text(path: Path) -> str:
    """Extract a PDF's full text, returning '' when it cannot be read."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return " ".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:  # a single unreadable file must not stop the build
        print(f"[registry] [warn] could not read '{path.name}': {exc}")
        return ""


def build_registry() -> dict:
    """Scan the PDF folders and return the registry structure."""
    models: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    applications: dict[str, set[str]] = defaultdict(set)
    canonical: dict[str, str] = {}
    catalogues: list[str] = []
    priority_docs: list[str] = []

    sources: list[tuple[Path, bool]] = []
    if DOCS_DIR.exists():
        sources += [(p, False) for p in dedupe_by_content(list(DOCS_DIR.glob("*.pdf")))]
    if XLC_DIR.exists():
        # The XLC engineering set supersedes the older per-model XLC datasheets.
        sources += [(p, True) for p in dedupe_by_content(list(XLC_DIR.glob("*.pdf")))]

    for path, is_priority in sources:
        stem = path.stem
        tokens = _tokenize_stem(stem)
        if not tokens:
            continue

        if _is_catalogue(stem):
            catalogues.append(path.name)
            continue

        if is_priority:
            priority_docs.append(path.name)

        family = tokens[0]
        families[family].add(path.name)

        for application in extract_applications(read_pdf_text(path)):
            applications[application].add(path.name)

        # Register every prefix of length >= 2 so 'fox 3f' and 'fox 3f rfp'
        # both resolve, and a user typing only the variant they know still hits.
        meaningful = [t for t in tokens if t not in _FAMILY_STOPWORDS]
        for length in range(2, len(meaningful) + 1):
            key = " ".join(meaningful[:length])
            models[key].add(path.name)

        # The document's own full code, as opposed to the prefixes above.
        # Prefix keys are deliberately shared — 'fox 3 f' resolves to all nine
        # FOX 3F documents so that a range question reaches them — but that left
        # no way to tell that FOX_3F.pdf *is* the FOX 3F while the others are its
        # variants. A question about the base model was therefore answered from
        # a variant's datasheet: the FOX 3F was reported at 64 bar, the pressure
        # of the carbon-steel FOX 3F-HP, while the base valve is ductile iron
        # PN 40. That is a pressure-containing part failing in service.
        canonical[" ".join(meaningful)] = path.name

    unify_translations(applications, [p for p, _ in sources])

    return {
        "_comment": (
            "Generated by ingest/build_model_registry.py. Maps valve model codes "
            "to datasheet filenames for exact lexical matching at query time."
        ),
        "models": {k: sorted(v) for k, v in sorted(models.items())},
        "canonical": dict(sorted(canonical.items())),
        "families": {k: sorted(v) for k, v in sorted(families.items())},
        "applications": {k: sorted(v) for k, v in sorted(applications.items())},
        "excluded": sorted(
            " ".join(_tokenize_stem(p.stem))
            for p in (EXCLUDED_DIR.glob("*.pdf") if EXCLUDED_DIR.exists() else [])
        ),
        "catalogues": sorted(catalogues),
        "priority_docs": sorted(priority_docs),
    }


def main() -> None:
    registry = build_registry()
    OUTPUT_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[registry] models     : {len(registry['models'])}")
    print(f"[registry] families   : {len(registry['families'])}")
    for name, files in registry["applications"].items():
        print(f"[registry]   application '{name}': {len(files)} datasheets")
    print(f"[registry] catalogues : {len(registry['catalogues'])}")
    print(f"[registry] priority   : {len(registry['priority_docs'])}")
    print(f"[registry] written to : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
