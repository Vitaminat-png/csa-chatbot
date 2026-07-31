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
    # La quota A della 150R arrivava dalla riga della 150 liscia (300 mm).
    ("Quanto pesa la SCS AS flangiata DN 150R e qual è la sua quota A?", ["29,7", "235"], ["300"]),
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


async def main() -> None:
    esiti = await asyncio.gather(*(chiedi(d) for d, _, _ in CASI))

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
