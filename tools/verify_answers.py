"""
Verifica i difetti confermati dall'audit multi-agente, con i valori attesi
presi dalle schede tecniche.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from openai import AsyncOpenAI

from api.links import sanitize_links
from api.models import HistoryMessage
from api.prompt import build_system_prompt
from api.retrieval import build_context_string, retrieve

# (domanda, deve contenere, non deve contenere)
CASI = [
    ("Qual e la pressione massima di esercizio della FOX 3F?", ["40"], ["64"]),
    ("Quanto pesa la FOX 3F flangiata DN 200?", ["85"], ["92"]),
    ("Quanto pesa la FOX 3F C flangiata DN 200?", ["92"], []),
    ("Quanto pesa lo sfiato LYNX 3F flangiato DN 200?", ["55"], ["60"]),
    ("Quanto pesa la XLC 400 DN 300 e le sue quote A e B?", ["405"], []),
    # Righe fuse da pdfplumber nelle tabelle CYCLOPS/GOLIA: la 150R leggeva il
    # 57 della 150 liscia da una riga doppia. Dopo lo split: 34.
    ("Quanto pesa la CYCLOPS 3F RFP flangiata DN 150R?", ["34"], ["57"]),
    ("Quanto pesa la GOLIA 3F flangiata DN 150R e qual è la sua quota A?", ["27", "235"], []),
    # Il pin per colonna seguiva l'ordine del pool: due tabelle di catalogo
    # davanti alla scheda del modello nominato, e la FOX SUB pesava 74 kg
    # (un altro prodotto) invece dei suoi 44,5.
    ("Quanto pesa la FOX SUB flangiata DN 150 e qual è la sua quota A?", ["44,5", "272"], ["74"]),
    # ATTENZIONE — questo caso conteneva una verita' sbagliata, corretta il
    # 30/08/2026. La scheda SCS_AS.pdf documenta la SCS-AS solo nella versione
    # 2" (p3: B 421 mm, 4 kg): non esiste una SCS-AS flangiata DN 150R. I
    # 29,7 kg stanno a p5, che e' il kit "GOLIA ... Mod. SUB" accodato allo
    # stesso file — un altro prodotto. Il bot deve rifiutare l'attribuzione,
    # e dare quel peso solo quando e' il kit a essere chiesto.
    ("Quanto pesa la SCS AS flangiata DN 150R?", [], ["29,7"]),
    ("Quanto pesa il kit SUB flangiato DN 150R?", ["29,7"], []),
    # La stessa domanda senza il contorno: era questa a fallire, perché "pesa"
    # non veniva riconosciuto come richiesta del peso. Il 304 escluso è il peso
    # del DN 300 della *serie 300*, la riga che finisce accanto a quella giusta.
    ("Quanto pesa la XLC 400 DN 300?", ["405"], ["304"]),
    ("How much does the XLC 400 DN 300 weigh?", ["405"], ["304"]),
    ("Quanto pesa una XLC 300 DN 400?", ["480"], ["704"]),
    ("Quali diametri nominali copre la serie XLC 400?", ["40", "800"], []),
    ("Quanti anni di garanzia offre CSA sulle sue valvole?", ["anno"], ["8 anni", "otto anni"]),
    ("La valvola ATHENA può lavorare a 25 bar?", ["16"], []),
    ("Qual è il Kv della XLC 400 versione standard DN 500?", ["non"], []),
    ("CSA è certificata secondo la UNI EN 558?", ["ISO 9001"], []),
    ("Quali valvole CSA sono adatte all'industria?", ["XLC"], []),
    ("Cosa proponete per un impianto di dissalazione?", ["GOLIA"], []),
    ("Quali prodotti CSA per impianti minerari?", ["GOLIA"], []),
    # Le dimensioni ITALICA esistono solo nelle pagine di famiglia del catalogo:
    # l'etichetta anti-variante le vietava (rifiuto), e senza etichette la
    # tabella FOX di p29 rispondeva 26 kg per colpa del "DN 100" nel heading
    # letto come serie. Il valore vero è 24,5 (corpo) / 27 (totale).
    ("Quanto pesa la ITALICA 310 DN 100?", [], ["26 kg"]),
    ("dimensioni della ITALICA 353", ["230", "165"], []),
    # Nomi storpiati: "atena" deve risolvere ATHENA, non rifiutare.
    ("mi dai le dimensioni di atena", ["230"], []),
    # --- Audit multi-agente del 30/08/2026: un caso per difetto confermato ---
    # Pressioni di serie spacciate per quelle della variante (25 bar su valvole
    # da 16): la classe piu' pericolosa trovata.
    ("Qual e la pressione massima di esercizio della XLC 353?", ["16"], ["25"]),
    ("Qual e la pressione massima di esercizio della XLC 380/480?", ["16"], ["25"]),
    ("Qual e la pressione massima di esercizio della XLC 310 ND?", ["16"], ["25"]),
    # Pagine "Working conditions": il dato c'e' ma il retrieval le perdeva.
    ("Qual e la pressione massima di esercizio della FOX 3F-C?", ["40"], []),
    ("Qual e la pressione massima della GOLIA 3F RFP?", ["40"], []),
    ("Qual e la pressione massima della SATURNO 3F RFP?", ["16"], []),
    ("Qual e la pressione minima di esercizio della XLC 321/421?", ["1,5"], []),
    ("Qual e la pressione massima di esercizio della XLC 365/465-MCP?", ["16"], []),
    ("Qual e la temperatura massima per la XLC 310/410-M?", ["70"], []),
    ("Qual e la pressione statica minima sul pilota della XLC 370/470-D?", ["0,3"], ["0,25"]),
    ("In quali classi di pressione PN e disponibile il serbatoio A.V.A.S.T.?", ["6"], []),
    # Quote: chiedere "quota A" non attivava il pin che i pesi attivavano.
    ("Qual e la quota A della XLC 330 DN 250?", ["730"], []),
    ("Qual e la quota C della XLC 330 DN 100?", ["118"], []),
    ("Quanto pesa la VRCA DN 200?", ["79"], []),
    # PDF multi-prodotto: il banner era per file, non per pagina.
    ("Quanto pesa la valvola SCS-AS?", ["4"], ["88,3"]),
    ("Qual e la quota B in mm della valvola SCS-AS 2 pollici?", ["421"], ["356"]),
    ("Qual e la quota A in mm dell'idrante a colonna Apollo RPC DN 80?", ["678"], ["682"]),
    ("Qual e la quota A in mm dell'idrante Apollo RP DN 80?", ["682"], []),
    ("Qual e la dimensione A massima del regolatore di flusso CSFL per valvole DN 80-100?", ["121"], []),
    # Serie 500/600: documentate a parte, e il catalogo le confondeva.
    ("Quanto pesa la XLC 600 DN 100?", ["43,5"], ["34"]),
    ("Quanto pesa la XLC 330 DN 80?", ["24"], ["20"]),
    ("Qual e la gamma di diametri DN disponibile per la serie XLC 500?", ["200"], []),
    ("In che materiale e la membrana del serbatoio SPT?", ["NBR"], ["EPDM"]),
    # --- Aggiornamento cataloghi del 02/09/2026 ---
    # Il Kv della Gemina DN 150 e' passato da 273 a 330 nei documenti 2026
    # (italiano e inglese concordi). Catalogo e scheda sono stati allineati
    # insieme: se uno solo dei due fosse aggiornato il bot risponderebbe
    # l'uno o l'altro a caso, ed e' proprio la classe di difetto che
    # l'audit aveva chiuso.
    ("Qual e il Kv della valvola Gemina DN 150?", ["330"], ["273"]),
    ("What is the Kv of the Gemina DN 150?", ["330"], ["273"]),
    # Schede aggiornate a giugno: controllo che restino leggibili dopo la
    # reingestione con le didascalie sopra le tabelle.
    ("Quanto pesa lo sfiato SCF flangiato DN 100?", ["40"], ["29"]),
    ("Qual e la pressione massima di esercizio della XLC 350?", ["16"], ["25"]),
    ("Quanto pesa la valvola Gemina FF?", ["2,3"], ["12"]),
    # Tre tabelle sulla stessa pagina, e la riga della 1"-1 1/4" non ha nemmeno
    # la colonna DN: senza la didascalia il modello leggeva un numero dal testo
    # speculare della barra laterale (12,6).
    ("Qual e la portata massima consigliata della valvola a galleggiante ATHENA 1 1/4?", ["1,9"], ["12,6"]),
    # --- Caratteristiche costruttive (02/09/2026) ---
    # "Avete valvole convogliate?" riceveva un rifiuto con quattro schede in
    # casa che lo sono: mancava l'indice per caratteristica, e recuperarle non
    # bastava — le schede sono in inglese ("air conveyance") e serviva dirlo
    # nel contesto anche col termine italiano.
    # Quale famiglia citi non conta — FOX, GOLIA, LYNX e SCF hanno tutte
    # modelli convogliati. Conta che non rifiuti.
    ("Avete valvole convogliate?", [], ["Non ho informazioni"]),
    ("Avete sfiati con scarico convogliato?", [], ["Non ho informazioni"]),
    ("Avete valvole anti colpo d'ariete?", [], ["Non ho informazioni"]),
    ("¿Cuál es la presión máxima de la ventosa FOX 3F?", ["40"], ["64"]),
    ("How much does the flanged DN 100 FOX 3F weigh?", ["21"], ["26"]),
]

LINGUE = [
    ("Cual es la presion maxima de trabajo de la valvula XLC 310?", "es"),
    ("Chi siamo e storia azienda", "it"),
    ("Vendete idranti antincendio?", "it"),
    ("How much does the FOX 3F weigh?", "en"),
]


async def chiedi(domanda: str, history=None) -> tuple[str, str]:
    srcs, lang = await retrieve(domanda, history=history)
    ctx = build_context_string(srcs, lang)
    oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await oai.chat.completions.create(
        model="gpt-4o-mini", max_tokens=800, temperature=0.2,
        messages=(
            [{"role": "system", "content": build_system_prompt(ctx, lang)}]
            + [{"role": h.role, "content": h.content} for h in (history or [])]
            + [{"role": "user", "content": domanda}]
        ),
    )
    allowed = []
    for s in srcs:
        if s.url:
            allowed.append(s.url)
        allowed.extend(s.url_alternates.values())
    return sanitize_links(resp.choices[0].message.content or "", allowed), lang


# Quanti casi far viaggiare insieme. La batteria e' cresciuta fino a superare
# il tetto di 200.000 token al minuto dell'account: lanciarli tutti in
# parallelo faceva fallire l'intera esecuzione con un 429, non un caso rosso.
CONCORRENZA = 6


async def chiedi_con_ripresa(domanda: str) -> tuple[str, str]:
    """Come `chiedi`, ma aspetta e riprova quando l'API impone il rate limit."""
    for tentativo in range(4):
        try:
            return await chiedi(domanda)
        except Exception as exc:
            if "rate_limit" not in str(exc).lower() or tentativo == 3:
                raise
            await asyncio.sleep(8 * (tentativo + 1))
    raise RuntimeError("irraggiungibile")


async def main() -> None:
    semaforo = asyncio.Semaphore(CONCORRENZA)

    async def limitato(domanda: str):
        async with semaforo:
            return await chiedi_con_ripresa(domanda)

    esiti = await asyncio.gather(*(limitato(d) for d, _, _ in CASI))

    ok = 0
    for (domanda, deve, non_deve), (risposta, _) in zip(CASI, esiti):
        problemi = [f"manca {v!r}" for v in deve if v.lower() not in risposta.lower()]
        problemi += [f"contiene {v!r}" for v in non_deve if v.lower() in risposta.lower()]
        if not problemi:
            ok += 1
        print(f"[{'OK  ' if not problemi else 'FAIL'}] {domanda[:62]}")
        for p in problemi:
            print(f"        !! {p} — risposta: {' '.join(risposta.split())[:150]}")

    print(f"\n=== dati tecnici: {ok}/{len(CASI)} ===")

    from api.retrieval import detect_language
    ok_lang = sum(1 for t, exp in LINGUE if detect_language(t) == exp)
    for t, exp in LINGUE:
        got = detect_language(t)
        print(f"[{'OK  ' if got == exp else 'FAIL'}] lingua {exp} -> {got}  {t[:50]}")
    print(f"=== lingua: {ok_lang}/{len(LINGUE)} ===")


if __name__ == "__main__":
    asyncio.run(main())
