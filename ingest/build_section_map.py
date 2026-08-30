"""
ingest/build_section_map.py
---------------------------
Genera `api/section_map.json`: per ogni scheda multi-prodotto, quale modello
documenta ciascuna pagina.

Alcuni PDF contengono piu' prodotti. APOLLO_RPC.pdf apre con "Mod. Apollo RP"
(p8) e prosegue con "Mod. Apollo RPC" (p10); SCS_AS.pdf accoda a p5 il kit
"GOLIA ... Mod. SUB". L'etichetta "questa e' LA scheda del modello chiesto" e'
per file, quindi finiva anche sulle pagine dell'altro prodotto: la quota A
dell'Apollo RPC veniva risposta con i 682 mm dell'Apollo RP, e il peso della
SCS-AS con i 7,0-88,3 kg del kit SUB.

Il marcatore "Mod. X" compare nella prosa di apertura di ogni sezione, mai nei
chunk-tabella, quindi va risolto a livello di documento: la dichiarazione a
pagina N vale fino alla dichiarazione successiva.

Uso:
  python -m ingest.build_section_map

Il file prodotto va committato: docs/ non entra nell'immagine Docker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = REPO / "api" / "section_map.json"
SERIES_OUT = REPO / "api" / "page_series.json"
XLC_DIR = REPO / "xlc engeniering"

# "Mod. Apollo RPC", "Mod. SUB", "Mod. XLC 310/410-ND". Si ferma a fine riga o
# a un connettivo di prosa: la dichiarazione e' un titolo, non una frase.
_MOD = re.compile(
    r"\bMod\.\s*([A-Z][A-Za-z0-9\"'/\.\- ]{1,40}?)"
    r"(?=\s*(?:$|\n|The |Il |La |is |e' |è |,|\.(?:\s|$)))",
    re.M,
)


def sections(pdf_path: Path) -> dict[str, str]:
    """{numero pagina: modello dichiarato} per ogni pagina del documento."""
    declared: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for pnum, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            # Solo la testa della pagina: "Mod. X" citato a meta' di un paragrafo
            # e' un riferimento incrociato, non l'apertura di una sezione.
            head = "\n".join(text.splitlines()[:6])
            found = _MOD.search(head)
            if found:
                name = " ".join(found.group(1).split()).rstrip(" .-")
                if name:
                    declared[pnum] = name

    if len(set(declared.values())) < 2:
        return {}  # documento a prodotto unico: nulla da disambiguare

    current = ""
    per_page: dict[str, str] = {}
    for pnum in range(1, total + 1):
        current = declared.get(pnum, current)
        if current:
            per_page[str(pnum)] = current
    return per_page


# Il catalogo generale dedica pagine distinte a ogni serie della stessa
# famiglia: la 298 e' la XLC 500, la 254 la XLC 400, la 423 la Italica 300.
# Sono tutte pagine "XLC" o "ITALICA" nei metadati, e trattarle come
# equivalenti diceva al modello che la tabella della XLC 500 rispondeva su una
# XLC 330: il suo DN 80 tornava 20 kg invece di 24. La serie non sta
# nell'intestazione — spesso e' un grafico — ma nel corpo della pagina.
_SERIES_IN_PAGE = re.compile(
    r"\b(XLC|Italica|FOX|GOLIA|LYNX|SCF|SCS|VRCD|VRCA|ARGO|ATHENA|SATURNO)"
    r"\s*(\d{3})\b",
    re.I,
)


def page_series(pdf_path: Path) -> dict[str, list[str]]:
    """{numero pagina: serie nominate nella pagina} per il catalogo generale."""
    per_page: dict[str, list[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            found = sorted({
                f"{m.group(1).lower()} {m.group(2)}"
                for m in _SERIES_IN_PAGE.finditer(text)
            })
            if found:
                per_page[str(pnum)] = found
    return per_page


def main() -> None:
    mapping: dict[str, dict[str, str]] = {}
    for pdf_path in sorted(DOCS.glob("*.pdf")):
        if pdf_path.name.startswith("Catalogo"):
            continue
        found = sections(pdf_path)
        if found:
            mapping[pdf_path.name] = found
            models = list(dict.fromkeys(found.values()))
            print(f"{pdf_path.name}: {len(found)} pagine, sezioni {models}")

    OUT.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\n{len(mapping)} schede multi-prodotto -> {OUT}")

    # Ogni PDF, non solo il catalogo: la serie di una pagina serve anche nelle
    # schede. In XLC_500_SIZING.pdf il chunk-tabella con i pesi della XLC 600
    # si intitola "[Table p.5]" e non nomina nulla — la serie sta nel chunk di
    # prosa accanto — quindi la selezione non poteva preferirlo a una pagina di
    # catalogo della XLC 500 e il peso del DN 100 restava senza risposta.
    series_map: dict[str, dict[str, list[str]]] = {}
    everything = sorted(DOCS.glob("*.pdf")) + sorted(XLC_DIR.glob("*.pdf"))
    for pdf_path in everything:
        found = page_series(pdf_path)
        if found:
            series_map[pdf_path.name] = found
    total = sum(len(v) for v in series_map.values())
    print(f"{total} pagine con una serie riconosciuta in {len(series_map)} documenti")
    SERIES_OUT.write_text(
        json.dumps(series_map, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"mappa pagina->serie -> {SERIES_OUT}")


if __name__ == "__main__":
    main()
