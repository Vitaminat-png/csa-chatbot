"""
ingest/render_dimension_pages.py
--------------------------------
Rende come PNG la pagina di ogni scheda che contiene la tabella dimensioni —
la stessa pagina porta la sagoma quotata che dà significato alle lettere
A/B/C/D/… — e scrive la mappa (source_file, page) -> immagine usata dall'API.

Una tabella di quote senza il disegno è una lista di numeri ciechi: il bot
elencava "A = 230 mm, B = 82,5 mm" e l'utente non aveva modo di sapere cosa
fossero A e B. La sagoma sta già nei PDF; da qui viene solo estratta.

Output:
  static/products/dimensions/<stem>.png       (una per scheda con tabella)
  api/dimension_drawings.json                 (mappa file+pagina -> url)

Uso:
  python -m ingest.render_dimension_pages

Da rieseguire quando si aggiunge o sostituisce una scheda in docs/. Le PNG
vanno committate: docs/ non entra nell'immagine Docker, quindi in produzione
le immagini esistono solo se stanno nel repo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

try:
    from ingest.pdf_extract import page_units
except ImportError:  # eseguito come script diretto
    from pdf_extract import page_units

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
XLC_DIR = REPO / "xlc engeniering"
OUT_DIR = REPO / "static" / "products" / "dimensions"
MAP_PATH = REPO / "api" / "dimension_drawings.json"

# La riga di una tabella dimensioni, in tutte le grafie del corpus.
WEIGHT_ROW = re.compile(r"(?:Weight|Wt|Peso|Poids)\s?\(?Kg\)?\s?=")

ZOOM = 2.0  # ~1190x1684 px per una pagina A4: leggibile ingrandita, file contenuto

# Le XLC engineering pubblicano le dimensioni una volta per serie; il disegno è
# identico nelle quattro edizioni, quindi si rende solo l'italiana e la mappa
# punta lì da ogni edizione.
XLC_SERIES_PAGES = {12: "xlc_serie_400", 20: "xlc_serie_300"}
XLC_EDITIONS = [
    "XLC engineering ITAL v2.pdf",
    "XLC engineering ENG v2.pdf",
    "XLC engineering FRA v2.pdf",
    "XLC engineering SPA v2.pdf",
]


def dimension_page(pdf_path: Path) -> int | None:
    """Pagina con più righe di tabella dimensioni, o None se non ce ne sono."""
    best_page, best_rows = None, 0
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            rows = sum(
                1
                for unit in page_units(page, pnum)
                for line in unit.splitlines()
                if WEIGHT_ROW.search(line)
            )
            if rows > best_rows:
                best_page, best_rows = pnum, rows
    return best_page


def render(pdf_path: Path, page_num: int, out_path: Path) -> None:
    with fitz.open(pdf_path) as doc:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        pix.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict[str, str]] = {}

    rendered = 0
    for pdf_path in sorted(DOCS.glob("*.pdf")):
        if pdf_path.name.startswith("Catalogo"):
            continue
        page_num = dimension_page(pdf_path)
        if page_num is None:
            continue
        out_path = OUT_DIR / f"{pdf_path.stem}.png"
        render(pdf_path, page_num, out_path)
        mapping.setdefault(pdf_path.name, {})[str(page_num)] = (
            f"/static/products/dimensions/{out_path.name}"
        )
        rendered += 1
        print(f"{pdf_path.name} p{page_num} -> {out_path.name}")

    # XLC engineering: un disegno per serie, mappato da tutte le edizioni.
    ital = XLC_DIR / "XLC engineering ITAL v2.pdf"
    if ital.exists():
        for page_num, stem in XLC_SERIES_PAGES.items():
            out_path = OUT_DIR / f"{stem}.png"
            render(ital, page_num, out_path)
            url = f"/static/products/dimensions/{out_path.name}"
            for edition in XLC_EDITIONS:
                mapping.setdefault(edition, {})[str(page_num)] = url
            rendered += 1
            print(f"{ital.name} p{page_num} -> {out_path.name} (tutte le edizioni)")

    MAP_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    total_kb = sum(f.stat().st_size for f in OUT_DIR.glob("*.png")) // 1024
    print(f"\n{rendered} immagini, {total_kb} KB totali -> {OUT_DIR}")
    print(f"mappa: {MAP_PATH}")


if __name__ == "__main__":
    main()
