"""
ingest/build_sizing_programs.py — indice dei programmi di calcolo CSA.

PERCHE' UN INDICE A MANO E NON IL CRAWL
Le pagine dei programmi di calcolo stanno quasi tutte dietro il login dell'area
clienti: a un visitatore anonimo rispondono 302 verso /login, quindi il crawler
ne indicizza solo un puntatore ("questa pagina esiste, serve accedere"). Da quel
puntatore non si ricava la cosa che al cliente serve davvero: **quali valvole
quel programma dimensiona**.

Il caso che ha fatto nascere questo file: a "c'e' un calcolatore per la valvola
AUGUSTA?" il bot rispondeva "non ho informazioni" e poi allegava comunque il
calcolatore XLC — un rifiuto e un link nella stessa risposta, per giunta il
programma di un'altra famiglia senza dire che era di un'altra famiglia.
Dimensionare una valvola col programma sbagliato e' un problema di sicurezza.

Il titolo della pagina nomina la sola serie ATHENA, ma il programma a
galleggiante dimensiona anche la AUGUSTA. Non e' deducibile da nessuna pagina
pubblica: e' un fatto dell'azienda, e sta scritto qui con la sua provenienza.
Riscontro indipendente: docs/AUGUSTA.pdf p2 rimanda a "the sizing calculator on
the CSA website" — l'unica scheda del corpus che citi un calcolatore.

COSA FA
1. Legge la sitemap live e tiene le pagine di dimensionamento.
2. Le raggruppa nei programmi che rappresentano, con le quattro lingue insieme.
3. Verifica una per una se rispondano al pubblico o rimandino al login: il
   crawler lo deduceva dalla lunghezza del corpo, e sbagliava su quattro pagine
   pubbliche il cui contenuto e' un iframe (corpo di testo quasi vuoto).
4. Scrive api/sizing_programs.json.

Uso:
    python -m ingest.build_sizing_programs
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
USCITA = REPO_ROOT / "api" / "sizing_programs.json"
SITEMAP = "https://csasrl.it/page-sitemap.xml"

# Una pagina che rimanda qui e' protetta, per quanto il suo corpo sia lungo.
SEGNO_DI_LOGIN = "/login"

# Slug che marcano una pagina come pagina di calcolo. Le quattro lingue usano
# parole diverse per la stessa sezione.
PAROLE_DI_CALCOLO = (
    "dimensionamento", "dimensionnement", "sizing", "programas-calculo",
    "programma-calcolo", "programma-per-calcolo", "calcolatore", "calculator",
    "calculateur", "calculadora", "-avs-",
)

# Sezioni del sito che parlano degli stessi prodotti senza essere programmi di
# calcolo. Senza questo filtro la parola "sfiati" faceva entrare due webinar,
# un video e una pagina di modelli 3D fra i programmi di dimensionamento:
# spacciare un webinar per un programma di calcolo e' una risposta falsa.
SEZIONI_NON_DI_CALCOLO = (
    "/modelli-3d", "/video-e-animazioni", "/webinar", "/3d-models",
    "/videos-and-animations", "/maquettes-3d", "/modelos-3d",
)

# ---------------------------------------------------------------------------
# I programmi, in ordine dal piu' specifico al piu' generico: il primo modello
# di slug che combacia vince, altrimenti "xlc" catturerebbe anche "xlc-400".
# ---------------------------------------------------------------------------
# `parole_tipo` sono le parole con cui un cliente nomina quel TIPO di valvola
# senza nominare una serie: "come dimensiono una valvola a galleggiante?" non
# contiene ne' ATHENA ne' AUGUSTA, ma chiede proprio il programma LVS.
# `famiglie` elenca le famiglie del registro modelli che quel programma
# dimensiona. E' la sola informazione che il bot non puo' ricavare da solo, ed
# e' quella che evita di indirizzare un cliente al programma sbagliato.
PROGRAMMI: list[dict] = [
    {
        "chiave": "cvs_xlc_400",
        "parole_tipo": r"pilotat|piloted|pilot[ée]e|automatic",
        "modelli_slug": ("xlc-400",),
        "sigla": "CSA CVS",
        "famiglie": ["XLC"],
        "nome": {
            "it": "CSA CVS per le valvole automatiche pilotate serie XLC 400",
            "en": "CSA CVS for XLC 400 series piloted automatic valves",
            "fr": "CSA CVS pour les vannes automatiques pilotées série XLC 400",
            "es": "CSA CVS para válvulas automáticas pilotadas de la serie XLC 400",
        },
        "copre": {
            "it": "valvole automatiche pilotate della serie XLC 400",
            "en": "XLC 400 series piloted automatic valves",
            "fr": "vannes automatiques pilotées de la série XLC 400",
            "es": "válvulas automáticas pilotadas de la serie XLC 400",
        },
    },
    {
        "chiave": "calcolatore_xlc",
        "parole_tipo": r"pilotat|piloted|pilot[ée]e|automatic",
        "modelli_slug": ("calcolatore-xlc", "xlc-series-calculator",
                         "calculateur-serie-xlc", "calculadora-de-la-serie-xlc"),
        "sigla": "",
        "famiglie": ["XLC"],
        "nome": {
            "it": "Calcolatore XLC (serie XLC 300 e XLC 400)",
            "en": "XLC Series Calculator (XLC 300 and XLC 400)",
            "fr": "Calculateur série XLC (XLC 300 et XLC 400)",
            "es": "Calculadora de la serie XLC (XLC 300 y XLC 400)",
        },
        "copre": {
            "it": "valvole automatiche pilotate delle serie XLC 300 e XLC 400",
            "en": "XLC 300 and XLC 400 series piloted automatic valves",
            "fr": "vannes automatiques pilotées des séries XLC 300 et XLC 400",
            "es": "válvulas automáticas pilotadas de las series XLC 300 y XLC 400",
        },
    },
    {
        "chiave": "cvs_xlc",
        "parole_tipo": r"pilotat|piloted|pilot[ée]e|automatic",
        "modelli_slug": ("cvs",),
        "sigla": "CSA CVS",
        "famiglie": ["XLC"],
        "nome": {
            "it": "CSA CVS per le valvole automatiche pilotate serie XLC",
            "en": "CSA CVS for XLC series piloted automatic valves",
            "fr": "CSA CVS pour les vannes automatiques pilotées série XLC",
            "es": "CSA CVS para válvulas automáticas pilotadas de la serie XLC",
        },
        "copre": {
            "it": "valvole automatiche pilotate della serie XLC",
            "en": "XLC series piloted automatic valves",
            "fr": "vannes automatiques pilotées de la série XLC",
            "es": "válvulas automáticas pilotadas de la serie XLC",
        },
    },
    {
        "chiave": "italica",
        "parole_tipo": r"pilotat|piloted|pilot[ée]e|automatic",
        "modelli_slug": ("italica",),
        "sigla": "",
        "famiglie": ["ITALICA"],
        "nome": {
            "it": "CSA per le valvole automatiche pilotate serie ITALICA",
            "en": "CSA for ITALICA series piloted automatic valves",
            "fr": "CSA pour les vannes automatiques pilotées série ITALICA",
            "es": "CSA para válvulas automáticas pilotadas serie ITALICA",
        },
        "copre": {
            "it": "valvole automatiche pilotate della serie ITALICA",
            "en": "ITALICA series piloted automatic valves",
            "fr": "vannes automatiques pilotées de la série ITALICA",
            "es": "válvulas automáticas pilotadas de la serie ITALICA",
        },
    },
    {
        # L'unico programma del sito che riguardi valvole a galleggiante. Il
        # titolo nomina la sola serie ATHENA; copre anche la AUGUSTA.
        "chiave": "lvs_galleggiante",
        "parole_tipo": r"galleggiant|flotteur|flotador|\bfloat\b|livello|\blevel\b|niveau|nivel|serbatoio|tank|r[ée]servoir|dep[oó]sito",
        "modelli_slug": ("lvs", "athena", "flotador", "flotteur", "float"),
        "sigla": "CSA LVS",
        "famiglie": ["ATHENA", "AUGUSTA"],
        "famiglie_fuori_titolo": ["AUGUSTA"],
        "provenienza": (
            "AUGUSTA non compare nel titolo della pagina: il programma sta "
            "dietro il login e la copertura e' stata confermata da CSA "
            "(04/09/2026). Riscontro: docs/AUGUSTA.pdf p2 rimanda al "
            "'sizing calculator on the CSA website'."
        ),
        "nome": {
            "it": "CSA LVS per le valvole a galleggiante (serie ATHENA e AUGUSTA)",
            "en": "CSA LVS for float valves (ATHENA and AUGUSTA series)",
            "fr": "CSA LVS pour les vannes à flotteur (séries ATHENA et AUGUSTA)",
            "es": "CSA LVS para válvulas de flotador (series ATHENA y AUGUSTA)",
        },
        "copre": {
            "it": "valvole a galleggiante delle serie ATHENA e AUGUSTA",
            "en": "ATHENA and AUGUSTA series float valves",
            "fr": "vannes à flotteur des séries ATHENA et AUGUSTA",
            "es": "válvulas de flotador de las series ATHENA y AUGUSTA",
        },
    },
    {
        "chiave": "rvs_gemina",
        "parole_tipo": r"azione diretta|direct[- ]acting|action directe|acci[oó]n directa|riduttric|riduzione di pressione|pressure reducing|r[ée]ductric|reductora",
        "modelli_slug": ("gemina", "rvs"),
        "sigla": "CSA RVS",
        "famiglie": ["GEMINA"],
        "nome": {
            "it": "CSA RVS per le valvole ad azione diretta serie GEMINA",
            "en": "CSA RVS for GEMINA series direct-acting valves",
            "fr": "CSA RVS pour les vannes à action directe série GEMINA",
            "es": "CSA RVS para válvulas de acción directa de la serie GEMINA",
        },
        "copre": {
            "it": "valvole ad azione diretta della serie GEMINA",
            "en": "GEMINA series direct-acting valves",
            "fr": "vannes à action directe de la série GEMINA",
            "es": "válvulas de acción directa de la serie GEMINA",
        },
    },
    {
        "chiave": "prs_vrcd",
        "parole_tipo": r"azione diretta|direct[- ]acting|action directe|acci[oó]n directa|riduttric|riduzione di pressione|pressure reducing|r[ée]ductric|reductora",
        "modelli_slug": ("vrcd", "prs"),
        "sigla": "CSA PRS",
        "famiglie": ["VRCD"],
        "nome": {
            "it": "CSA PRS per le valvole ad azione diretta serie VRCD e VRCD M",
            "en": "CSA PRS for VRCD and VRCD M series direct-acting valves",
            "fr": "CSA PRS pour les vannes à action directe séries VRCD et VRCD M",
            "es": "CSA PRS para válvulas de acción directa series VRCD y VRCD M",
        },
        "copre": {
            "it": "valvole ad azione diretta delle serie VRCD e VRCD M",
            "en": "VRCD and VRCD M series direct-acting valves",
            "fr": "vannes à action directe des séries VRCD et VRCD M",
            "es": "válvulas de acción directa de las series VRCD y VRCD M",
        },
    },
    {
        # Il titolo non nomina alcuna famiglia, e nessuna pagina pubblica dice
        # quali sfiati copra: qui non se ne dichiara nessuna. Un elenco
        # inventato manderebbe un cliente a dimensionare lo sfiato sbagliato.
        "chiave": "avs_sfiati",
        "parole_tipo": r"sfiat|ventos|ventouse|air valve|air release|espulsione dell.aria|expulsi[oó]n de aire|purgeur|degas",
        "modelli_slug": ("avs", "programma-calcolo-sfiati",
                         "programma-per-calcolo-sfiati", "air-release",
                         "expulsion-de-aire"),
        "sigla": "CSA AVS",
        "famiglie": [],
        "nome": {
            "it": "Programma di calcolo degli sfiati per una condotta (CSA AVS)",
            "en": "CSA AVS pipeline air release valve sizing program",
            "fr": "Programme de dimensionnement des ventouses pour conduite (CSA AVS)",
            "es": "Programa de dimensionamiento de válvulas de expulsión de aire (CSA AVS)",
        },
        "copre": {
            "it": "il dimensionamento delle valvole di sfiato su una condotta "
                  "(la pagina non dichiara a quali serie si applichi)",
            "en": "sizing of air release valves along a pipeline "
                  "(the page does not state which series it applies to)",
            "fr": "le dimensionnement des ventouses sur une conduite "
                  "(la page n'indique pas à quelles séries elle s'applique)",
            "es": "el dimensionamiento de las válvulas de expulsión de aire en "
                  "una tubería (la página no indica a qué series se aplica)",
        },
    },
]

# La pagina indice della sezione: utile come punto di partenza, non e' un
# programma e non dimensiona nulla.
SLUG_INDICE = ("documentazione/dimensionamento/", "en/documentation/sizing/",
               "fr/documentation/dimensionnement/", "es/documentacion/programas-calculo/")


def lingua_di(url: str) -> str:
    percorso = urlparse(url).path.lstrip("/")
    for codice in ("en", "fr", "es"):
        if percorso.startswith(codice + "/"):
            return codice
    return "it"


def e_pagina_di_calcolo(url: str) -> bool:
    percorso = urlparse(url).path.lower()
    if any(sezione in percorso for sezione in SEZIONI_NON_DI_CALCOLO):
        return False
    return any(parola in percorso for parola in PAROLE_DI_CALCOLO)


def e_indice(url: str) -> bool:
    percorso = urlparse(url).path.lstrip("/").lower()
    return percorso in SLUG_INDICE


def programma_di(url: str) -> dict | None:
    """Il primo programma il cui modello di slug compaia nell'URL."""
    percorso = urlparse(url).path.lower()
    for programma in PROGRAMMI:
        if any(modello in percorso for modello in programma["modelli_slug"]):
            return programma
    return None


def stato_accesso(client: httpx.Client, url: str) -> str:
    """
    'pubblica', 'login' o 'assente'.

    Il criterio e' la destinazione del redirect, non la lunghezza del corpo:
    i calcolatori pubblici XLC e VRCD mostrano un iframe, quindi hanno un corpo
    di testo quasi vuoto e il criterio della lunghezza li dichiarava protetti.
    """
    try:
        risposta = client.get(url, timeout=25.0, follow_redirects=False)
    except httpx.HTTPError as errore:
        print(f"[sizing]   irraggiungibile {url}: {errore}")
        return "sconosciuto"
    destinazione = risposta.headers.get("location", "")
    if risposta.status_code in (301, 302, 303, 307, 308):
        return "login" if SEGNO_DI_LOGIN in destinazione else "pubblica"
    if risposta.status_code >= 400:
        return "assente"
    return "pubblica"


def main() -> None:
    with httpx.Client(headers={"User-Agent": "csa-chatbot/1.0"}) as client:
        xml = client.get(SITEMAP, timeout=30.0, follow_redirects=True).text
        tutte = re.findall(r"<loc>([^<]+)</loc>", xml)
        pagine = sorted({u for u in tutte if e_pagina_di_calcolo(u)})
        print(f"[sizing] {len(pagine)} pagine di calcolo nella sitemap")

        raccolta: dict[str, dict] = {}
        indice: list[dict] = []
        orfane: list[str] = []

        for url in pagine:
            accesso = stato_accesso(client, url)
            if accesso == "assente":
                # Nella sitemap del sito c'e' un URL rotto (manca il prefisso di
                # lingua): indicizzarlo manderebbe i clienti su un 404.
                print(f"[sizing]   404, scartata: {url}")
                continue
            voce = {"url": url, "lingua": lingua_di(url), "accesso": accesso}
            if e_indice(url):
                indice.append(voce)
                continue
            programma = programma_di(url)
            if programma is None:
                orfane.append(url)
                continue
            dato = raccolta.setdefault(programma["chiave"], {
                "chiave": programma["chiave"],
                "sigla": programma["sigla"],
                "famiglie": programma["famiglie"],
                "nome": programma["nome"],
                "copre": programma["copre"],
                "parole_tipo": programma.get("parole_tipo", ""),
                "pagine": [],
            })
            for campo in ("famiglie_fuori_titolo", "provenienza", "parole_tipo"):
                if campo in programma:
                    dato[campo] = programma[campo]
            dato["pagine"].append(voce)

    for url in orfane:
        print(f"[sizing]   nessun programma riconosciuto: {url}")

    programmi = [raccolta[p["chiave"]] for p in PROGRAMMI if p["chiave"] in raccolta]
    for dato in programmi:
        dato["pagine"].sort(key=lambda v: (v["lingua"], v["url"]))
        pubbliche = sum(1 for v in dato["pagine"] if v["accesso"] == "pubblica")
        print(f"[sizing] {dato['chiave']:<18} {len(dato['pagine'])} pagine "
              f"({pubbliche} pubbliche)  famiglie={dato['famiglie'] or '-'}")

    USCITA.write_text(json.dumps({
        "_comment": (
            "Generato da ingest/build_sizing_programs.py. I programmi di calcolo "
            "CSA stanno quasi tutti dietro il login, quindi il crawler ne vede "
            "solo un puntatore: quali valvole ciascuno dimensioni sta scritto "
            "qui. Lo stato di accesso e' verificato sul redirect, non sulla "
            "lunghezza del corpo."
        ),
        "programmi": programmi,
        "indice_sezione": sorted(indice, key=lambda v: v["lingua"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sizing] scritto {USCITA} — {len(programmi)} programmi, "
          f"{sum(len(d['pagine']) for d in programmi)} pagine")


if __name__ == "__main__":
    main()
