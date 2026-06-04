# CSA AI Chatbot - Documentazione Progetto

## 1. Panoramica Progetto

- **Nome:** CSA AI Chatbot
- **Scopo:** Chatbot per il sito csasrl.it che risponde a domande sui prodotti CSA in qualsiasi lingua
- **Architettura:** RAG (Retrieval Augmented Generation) con Pinecone + GPT-4o mini + FastAPI
- **Costo operativo:** ~8 euro/mese (OpenAI API usage) + Render hosting
- **URL live:** https://csa-chatbot.onrender.com/
- **Repository:** https://github.com/Vitaminat-png/csa-chatbot

---

## 2. Architettura Tecnica

- **Pinecone (free tier Starter):** database vettoriale, indice `csa-chatbot`, 1536 dimensioni, metrica cosine
- **OpenAI GPT-4o mini:** genera risposte in qualsiasi lingua
- **OpenAI text-embedding-3-small:** genera embedding per i chunk di testo
- **FastAPI:** backend API con endpoint `/api/chat` e `/api/chat/stream` (SSE)
- **Widget HTML/JS:** interfaccia chat servita direttamente dalla root `/`
- **Hosting:** Render.com (piano Free, region Frankfurt)

---

## 3. Dati Ingeriti

- **115 PDF schede tecniche inglesi** da `Schede Tecniche/EN/` → 521 vettori
- **936 URL scrappati** dai sitemap di csasrl.it in 4 lingue IT/EN/FR/ES → 935 vettori
- **Totale:** 1456 vettori in Pinecone
- I PDF sono **SOLO in inglese**; il modello traduce le risposte nella lingua dell'utente
- `url_map.json` contiene la mappatura URL per lingua per ogni prodotto

---

## 4. Decisioni Chiave

- Ingestione solo documenti EN (LLM traduce)
- Scraping sitemaps con `httpx` + `BeautifulSoup` (non Playwright)
- Pinecone batch size 20 (free tier ha timeout su batch grandi)
- Filtro URL: il bot non suggerisce mai pagine `/shop/`, `/cart/`, `/checkout/` ecc.
- CORS configurato per csasrl.it e localhost
- Widget usa URL relativi (funziona sia in locale che su Render)
- `.env` NON committato su GitHub (le chiavi sono nelle Environment Variables di Render)

---

## 5. Credenziali e Configurazione

- **OpenAI API Key:** configurata su Render env vars (creata dall'utente su platform.openai.com)
- **Pinecone API Key:** configurata su Render env vars (piano Starter free, app.pinecone.io)
- **Pinecone Index:** `csa-chatbot` (creato manualmente dall'utente)
- **Piano Render:** Free (si addormenta dopo 15 min, 30s cold start)

---

## 6. Struttura File

```
csa-chatbot/
├── api/
│   ├── main.py        (FastAPI app, GET /, POST /api/chat, /api/chat/stream)
│   ├── models.py      (Pydantic schemas)
│   ├── prompt.py      (system prompt template)
│   └── retrieval.py   (Pinecone query + URL filtering)
├── ingest/
│   ├── pdf_ingest.py  (pdfplumber chunking + embedding + upsert)
│   ├── web_scraper.py (sitemap crawl + URL mapping + upsert)
│   └── run_all.py     (orchestrator)
├── tests/
│   └── test_20_questions.py  (25 test, 19 unit + 6 integration)
├── widget/
│   └── chatbot.html   (chat UI con branding CSA)
├── docs/              (115 PDF EN - non su GitHub)
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .env               (locale, non su GitHub)
├── env.example
└── .gitignore
```

---

## 7. Come Aggiungere Nuovi Documenti

1. Mettere i PDF inglesi nella cartella `docs/`
2. Eseguire:
   ```bash
   python -m ingest.run_all
   ```
3. I nuovi vettori vengono aggiunti a Pinecone automaticamente

---

## 8. Come Aggiornare gli URL del Sito

Se cambiano le pagine su csasrl.it:

1. Eseguire:
   ```bash
   python -m ingest.web_scraper
   ```
2. Questo rigenera `url_map.json` e aggiorna i vettori URL in Pinecone

---

## 9. Problemi Noti e Soluzioni

| Problema | Soluzione |
|----------|-----------|
| Pinecone free tier timeout | Batch size ridotto a 20, timeout 120s, retry con backoff |
| csasrl.it blocca download immagini (403) | Usato estrazione da PDF con PyMuPDF |
| PowerShell UTF-8 BOM | Usato `System.IO.File.WriteAllText` con `UTF8Encoding(false)` |
| Render Free cold start | ~30s dopo 15 min inattività. Upgrade a Starter (7 USD/mese) per sempre attivo |

---

## 10. Prossimi Passi Possibili

- Integrare widget nel sito csasrl.it (aggiungere script tag nel tema WordPress)
- Upgrade Render a Starter per eliminare cold start
- Aggiungere catalogo generale PDF (non solo schede tecniche)
- Implementare feedback utente (thumbs up/down)
- Analytics: tracciare domande più frequenti
