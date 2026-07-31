"""
probe.py — interroga il chatbot CSA e restituisce tutto il necessario per giudicarne la risposta.

Uso:
    python probe.py "domanda"
    python probe.py "domanda1" "domanda2" ...
    python probe.py --json "domanda"
    python probe.py --chat "prima domanda" "follow-up" ...   # turni consecutivi, con memoria
    python probe.py --context "domanda"                      # stampa anche il contesto completo

Per ogni domanda stampa:
  - lingua rilevata
  - fonti recuperate (file, pagina, punteggio, applicazioni documentate, URL)
  - risposta
  - link presenti nella risposta, con verifica: nel contesto? raggiungibile in url_map?
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from openai import AsyncOpenAI

from api.links import sanitize_links
from api.models import HistoryMessage
from api.prompt import build_system_prompt
from api.retrieval import build_context_string, retrieve

LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")

_url_map = json.loads((REPO / "url_map.json").read_text(encoding="utf-8"))
REACHABLE = {u.rstrip("/") + "/" for u in _url_map}
for _v in _url_map.values():
    for _t in _v.values():
        REACHABLE.add(_t.rstrip("/") + "/")


async def ask(question: str, history: list | None = None) -> dict:
    srcs, lang = await retrieve(question, history=history)
    ctx = build_context_string(srcs, lang)
    oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await oai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=900,
        temperature=0.2,
        messages=(
            [{"role": "system", "content": build_system_prompt(ctx, lang)}]
            + [{"role": h.role, "content": h.content} for h in (history or [])]
            + [{"role": "user", "content": question}]
        ),
    )
    raw = resp.choices[0].message.content or ""
    # Stessa sanificazione dei link applicata dall'API in produzione.
    answer = sanitize_links(raw, [s.url for s in srcs if s.url])

    links = []
    for url in LINK_RE.findall(answer):
        links.append({
            "url": url,
            "nel_contesto": url in ctx,
            "raggiungibile": url.rstrip("/") + "/" in REACHABLE,
        })

    return {
        "domanda": question,
        "lingua": lang,
        "fonti": [
            {
                "file": s.source_file,
                "pagina": s.page,
                "punteggio": s.score,
                "titolo": s.page_title,
                "applicazioni": s.applications,
                "url": s.url,
                "estratto": s.text_full[:400],
            }
            for s in srcs
        ],
        "risposta": answer,
        "risposta_grezza": raw,
        "link": links,
        "contesto": ctx,
    }


def stampa(r: dict, con_contesto: bool = False) -> None:
    print("=" * 78)
    print(f"[{r['lingua']}] {r['domanda']}")
    print("-" * 78)
    print("FONTI:")
    for f in r["fonti"]:
        app = f" | applicazioni: {', '.join(f['applicazioni'])}" if f["applicazioni"] else ""
        nome = f["file"] if f["file"] != "csasrl.it" else (f["titolo"] or "")[:52]
        print(f"  {f['punteggio']:.3f}  {nome} p{f['pagina']}{app}")
    print("RISPOSTA:")
    print(r["risposta"])
    if r["link"]:
        print("LINK:")
        for l in r["link"]:
            flag = "ok" if l["nel_contesto"] and l["raggiungibile"] else "SOSPETTO"
            print(f"  [{flag}] {l['url']}  (nel contesto: {l['nel_contesto']}, "
                  f"raggiungibile: {l['raggiungibile']})")
    if con_contesto:
        print("-" * 78)
        print("CONTESTO COMPLETO:")
        print(r["contesto"])
    print()


async def main() -> None:
    args = [a for a in sys.argv[1:]]
    as_json = "--json" in args
    chat_mode = "--chat" in args
    con_contesto = "--context" in args
    domande = [a for a in args if not a.startswith("--")]

    if not domande:
        print(__doc__)
        return

    risultati = []
    if chat_mode:
        history: list[HistoryMessage] = []
        for d in domande:
            r = await ask(d, history=history)
            risultati.append(r)
            history.append(HistoryMessage(role="user", content=d))
            history.append(HistoryMessage(role="assistant", content=r["risposta"]))
    else:
        risultati = list(await asyncio.gather(*(ask(d) for d in domande)))

    if as_json:
        print(json.dumps(risultati, ensure_ascii=False, indent=2))
    else:
        for r in risultati:
            stampa(r, con_contesto)


if __name__ == "__main__":
    asyncio.run(main())
