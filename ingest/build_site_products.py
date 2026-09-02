"""
ingest/build_site_products.py
-----------------------------
Genera `api/site_products.json`: quali pagine prodotto del sito stanno in
quale categoria.

Serve per i prodotti che il sito pubblica ma le schede tecniche non
documentano — o non devono documentare. L'elenco che il bot produce a una
domanda tipo "quali idranti fate?" si costruisce dall'indice delle schede: un
prodotto senza scheda indicizzata semplicemente non compariva, anche se la sua
pagina era nell'indice e rispondeva benissimo quando lo si nominava.

Le categorie stanno in api/site_structure.json; i prodotti si leggono dalla
pagina di categoria stessa. I chunk del sito portano il `canonical_url`
italiano anche nelle edizioni straniere, quindi la mappa in italiano copre
tutte e quattro le lingue.

Uso:
  python -m ingest.build_site_products

Va rieseguito quando si pubblica o si sposta un prodotto, come il crawler.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parent.parent
STRUCTURE = REPO / "api" / "site_structure.json"
URL_MAP = REPO / "url_map.json"
OUT = REPO / "api" / "site_products.json"

# Il sito risponde 403 a un client senza intestazioni da browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
PAUSA = 0.8   # crawl gentile, come site_crawler

_PRODOTTO = re.compile(r"/(?:prodotto|product|productos|produit)/", re.I)


def categorie_italiane() -> list[str]:
    """Gli URL delle pagine di categoria in italiano."""
    dati = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    sezioni = dati.get("pages_by_section", {})
    urls: list[str] = []
    for nome, per_lingua in sezioni.items():
        if "categoria-prodotto" not in nome:
            continue
        if isinstance(per_lingua, dict):
            urls += [u for u in per_lingua.get("it", []) if "/categoria-prodotto/" in u]
    return sorted(set(urls))


def prodotti_di(url: str, client: httpx.Client) -> list[dict]:
    """Le pagine prodotto elencate in una pagina di categoria."""
    try:
        r = client.get(url, timeout=30, follow_redirects=True)
        r.raise_for_status()
    except Exception as exc:
        print(f"  [avviso] {url}: {exc}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    visti: dict[str, str] = {}
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").split("?")[0]
        if not _PRODOTTO.search(href) or href in visti:
            continue
        titolo = a.get_text(" ", strip=True)
        # Il link sull'immagine e' vuoto: il titolo arriva dal link gemello.
        visti[href] = titolo
    return [
        {"url": u, "titolo": t}
        for u, t in visti.items()
        if t and len(t) > 3
    ]


def main() -> None:
    mappa: dict[str, dict] = {}
    url_map = json.loads(URL_MAP.read_text(encoding="utf-8"))
    with httpx.Client(headers=HEADERS) as client:
        for url in categorie_italiane():
            slug = url.rstrip("/").rsplit("/", 1)[-1]
            prodotti = prodotti_di(url, client)
            if prodotti:
                # Gli slug nelle quattro lingue: una domanda inglese
                # ("which hydrants do you make?") non incontrerebbe mai la
                # parola italiana "idranti".
                alternative = url_map.get(url, {})
                slugs = sorted({
                    u.rstrip("/").rsplit("/", 1)[-1]
                    for u in [url, *alternative.values()]
                })
                mappa[slug] = {
                    "categoria": url,
                    "slug_lingue": slugs,
                    "prodotti": prodotti,
                }
                print(f"{slug}: {len(prodotti)} prodotti | slug {slugs}")
            time.sleep(PAUSA)

    OUT.write_text(
        json.dumps(mappa, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    totale = sum(len(v["prodotti"]) for v in mappa.values())
    print(f"\n{len(mappa)} categorie, {totale} pagine prodotto -> {OUT}")


if __name__ == "__main__":
    main()
