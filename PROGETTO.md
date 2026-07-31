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

Quattro sorgenti, tutte ricostruibili con `python -m ingest.run_all`:

| Sorgente | Script | Namespace | Contenuto |
|----------|--------|-----------|-----------|
| Schede tecniche PDF (115 file) | `ingest/pdf_ingest.py` | default | Schede prodotto in inglese |
| Catalogo generale | `ingest/catalog_ingest.py` | `catalog` | Catalogo italiano con metadati per famiglia |
| XLC engineering (IT/EN/FR/ES) | `ingest/xlc_ingest.py` | default | Riferimento aggiornato su serie XLC 300/400 |
| Contenuto sito csasrl.it | `ingest/site_crawler.py` | default | Testo delle pagine del sito, in 4 lingue |

Il catalogo è escluso da `pdf_ingest` anche se sta in `docs/`: lo gestisce
`catalog_ingest.py`, che lo indicizza nel namespace `catalog` con i metadati di
sezione e famiglia. Processarlo in entrambi produceva una seconda copia peggiore
delle stesse 400+ pagine.

Tutti gli script PDF condividono `ingest/pdf_extract.py`, che serializza le
tabelle e ripulisce il rumore dei grafici — vedi le decisioni qui sotto.

- I PDF delle schede sono **solo in inglese**; il modello traduce.
- Le XLC engineering esistono in 4 lingue: ogni chunk ha metadato `lang` e il
  retrieval preferisce l'edizione nella lingua dell'utente.
- `url_map.json` contiene solo URL **raggiungibili navigando il sito**.
- `api/model_registry.json` mappa i codici modello ai file scheda.
- `api/site_structure.json` è l'inventario delle pagine raggiungibili per sezione.

---

## 4. Decisioni Chiave

- Ingestione documenti EN per le schede (LLM traduce); XLC engineering in 4 lingue
  perché è il riferimento su taglie e materiali e la terminologia nativa evita
  che una risposta spagnola contenga termini italiani.
- Scraping con `httpx` + `BeautifulSoup` (non Playwright): le pagine sono
  renderizzate lato server.
- Pinecone batch size 20 (free tier ha timeout su batch grandi).
- Filtro URL: il bot non suggerisce mai pagine `/shop/`, `/negozio/`, `/cart/`,
  `/checkout/`, account/login, né gli archivi tag `/tag-product/`.
- **Pagine del sito, tre categorie.** L'indice parte da un attraversamento in
  ampiezza del grafo dei link dalle 4 homepage di lingua. Poi le pagine del
  sitemap che il crawl non ha raggiunto vengono verificate una a una:
  - corpo praticamente vuoto → è l'**area riservata** (i programmi di calcolo
    CSA CVS/AVS/RVS/PRS/LVS). Indicizzate come puntatore: cosa sono e che serve
    il login. Il contenuto dietro il login non viene mai letto.
  - slug con `bozza`/`prova`/`draft` → **scartate** (es. `/en/home-eng-bozza/`,
    `/prova-menu-prodotti/`).
  - contenuto reale ma nessun link entrante → **indicizzate**: i calcolatori
    pubblici Italica/Athena/Gemina funzionano, sono solo slegati dal menu.
- **Tabelle serializzate** (`ingest/pdf_extract.py`): ogni tabella diventa una
  riga autonoma (`DN (mm) = 800; Kv (m3/h) = 10479`). `extract_text()` le
  appiattiva perdendo l'associazione taglia→valore e omettendo l'ultima colonna:
  su un campione di 120 chunk delle schede, l'11% era per oltre un quinto cifre.
- **Rumore dei grafici rimosso**: le etichette degli assi delle curve di
  prestazione finiscono nel livello testo come sequenze di numeri nudi.
- **Match esatto sui codici modello**: gli embedding non distinguono
  "ITALICA 353" da "ITALICA 310", quindi una query che nomina un modello
  interroga anche direttamente la scheda di quel modello.
- **Espansione della query in inglese**: le 115 schede sono solo in inglese e
  perdevano contro il catalogo italiano su ogni domanda in italiano
  (`per irrigazione cosa consigli?` metteva ARGO.pdf 8º a 0.390, mentre
  `valves for irrigation` lo metteva 1º a 0.543). Ogni domanda non inglese viene
  cercata anche in traduzione inglese.
- **Tetto per documento**: nessun documento può occupare più di 4 slot fra i
  candidati. Il catalogo ha 2155 chunk contro ~800 delle schede e da solo ne
  prendeva 16 su 20.
- **Indice per applicazione**: `api/model_registry.json` mappa anche le
  applicazioni (irrigazione, fognatura, antincendio, acqua di mare…) ai file
  scheda che le documentano. A una domanda tipo "per irrigazione cosa consigli?"
  non si può rispondere con la sola similarità fra passaggi: la risposta è un
  *insieme* di prodotti, e 32 schede citano l'irrigazione mentre il bot ne
  elencava due. Il contesto riporta per ogni scheda le applicazioni che
  documenta, così il modello cita anche i prodotti il cui estratto parla d'altro.
- **Follow-up con memoria**: un messaggio breve come "e basta?" non nomina
  nulla, quindi il retrieval non trovava niente e il bot rispondeva "non ho
  informazioni" subito dopo aver risposto sul tema. I follow-up brevi vengono
  cercati insieme al turno utente precedente.
- **Modello base ≠ variante**: i suffissi (-HP, -AS, -C, -RFP…) identificano
  prodotti diversi. `api/model_registry.json` contiene una mappa `canonical`
  (codice completo → scheda), e le chiavi si cercano come **token contigui**:
  senza questo "XLC 400 DN 300" veniva letto come modello "XLC 300", e la FOX 3F
  riceveva i 64 bar della FOX 3F-HP (corpo in acciaio) invece dei suoi 40 bar su
  corpo in ghisa — un componente in pressione portato oltre il suo limite.
  La scheda del modello nominato apre il contesto ed è etichettata; le altre sono
  etichettate come varianti, perché etichettare solo quella giusta non bastava
  (il valore sbagliato compariva due volte e il modello seguiva la maggioranza).
- **Chunk adiacenti**: a ogni fonte scelta vengono accodati i chunk vicini della
  stessa pagina, recuperati per id. Il taglio in chunk spezza le frasi a metà:
  alla domanda sulla garanzia arrivava la clausola "entro 8 giorni dalla scoperta
  del difetto" ma non la durata, e il bot rispondeva "8 anni" invece di un anno.
- **Sonda per tabelle**: una domanda su una taglia viene cercata anche nella
  forma delle tabelle serializzate (`DN (mm) = 300; …`), perché una frase in
  prosa somiglia poco a una riga di numeri.
- **Fallback per lingua**: il rifiuto era una frase italiana che il modello
  doveva "tradurre se necessario", e spesso non lo faceva. Ora ogni lingua ha la
  sua, e per le lingue fuori tabella il modello scrive in quella dell'utente.
- **Lingua ereditata dalla conversazione**: un follow-up breve ("e basta?") non
  ha segnale e ricadeva sull'inglese, ribaltando la lingua a metà dialogo. Ora
  eredita dal turno precedente, e un pareggio non viene più risolto per ordine
  del dizionario (che premiava sempre l'italiano).
- **Il contesto sta in fondo al prompt**, dopo tutte le regole: quando stava in
  mezzo, le regole che lo seguivano venivano ignorate (il bot inventava il Kv di
  una taglia inesistente). Il blocco dichiara anche che il testo recuperato è
  **dato, non istruzione** — le pagine del sito finiscono nel prompt.
- **Validazione dei link** (`api/links.py`): ogni link della risposta viene
  confrontato con gli URL effettivamente recuperati; se non corrisponde viene
  corretto o il link viene rimosso lasciando il testo. Il prompt da solo non
  bastava — a una domanda in spagnolo il modello "correggeva" lo slug italiano
  `alta-efficienza` in `alta-efficiencia`, producendo un 404, in modo
  intermittente. Funziona anche in streaming, trattenendo solo i frammenti di
  link.
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
│   ├── main.py             (FastAPI app, GET /, POST /api/chat, /api/chat/stream)
│   ├── models.py           (Pydantic schemas)
│   ├── prompt.py           (system prompt template)
│   ├── retrieval.py        (query Pinecone + rerank + lingua + filtro URL)
│   ├── model_index.py      (match dei codici modello e delle applicazioni)
│   ├── links.py            (verifica i link della risposta, anche in streaming)
│   ├── model_registry.json (generato: codice modello -> file scheda)
│   └── site_structure.json (generato: inventario pagine raggiungibili)
├── ingest/
│   ├── build_model_registry.py (genera api/model_registry.json)
│   ├── pdf_extract.py      (estrazione PDF condivisa: tabelle + chunking)
│   ├── pdf_ingest.py       (schede docs/ -> embedding + upsert)
│   ├── catalog_ingest.py   (catalogo generale -> namespace 'catalog')
│   ├── xlc_ingest.py       (XLC engineering 4 lingue, tabelle serializzate)
│   ├── site_crawler.py     (crawl del grafo dei link + contenuto pagine)
│   ├── web_scraper.py      (SUPERATO da site_crawler.py — non eseguire)
│   └── run_all.py          (orchestratore dei 4 step)
├── tests/
│   └── test_20_questions.py  (35 test, 29 unit + 6 integration)
├── pytest.ini              (marker 'integration', modalità asyncio)
├── widget/
│   └── chatbot.html        (chat UI con branding CSA)
├── docs/                   (116 PDF EN - non su GitHub)
├── xlc engeniering/        (4 PDF XLC in IT/EN/FR/ES - non su GitHub)
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
2. Eseguire l'intera pipeline:
   ```bash
   python -m ingest.run_all
   ```
3. I nuovi vettori vengono aggiunti a Pinecone automaticamente

Per rigenerare **solo** il registro dei modelli (dopo aver aggiunto o rinominato
una scheda in `docs/`, senza rifare l'ingestione):

```bash
python -m ingest.build_model_registry
```

Nota: `api/model_registry.json` va committato — `docs/` non finisce nell'immagine
Docker, quindi in produzione il registro è l'unica fonte per il match sui modelli.

---

## 8. Come Aggiornare i Contenuti del Sito

Se cambiano le pagine su csasrl.it:

```bash
python -m ingest.site_crawler
```

Lo script:
1. legge le varianti di lingua dal sitemap (`sitemap_index.xml`);
2. attraversa il grafo dei link partendo dalle 4 homepage di lingua;
3. riscrive `url_map.json` e `api/site_structure.json` con le sole pagine raggiungibili;
4. cancella i vettori del run precedente e reindicizza il contenuto delle pagine.

Va rieseguito quando si pubblicano o si rinominano pagine: le pagine rimosse dal
sito spariscono dall'indice al run successivo. Richiede ~5 minuti (crawl gentile,
4 richieste concorrenti con pausa).

**Non eseguire `ingest/web_scraper.py`**: è la versione superata e reintroduce i
vettori delle pagine orfane.

---

## 9. Problemi Noti e Soluzioni

| Problema | Soluzione |
|----------|-----------|
| Pinecone free tier timeout | Batch size ridotto a 20, timeout 120s, retry con backoff |
| csasrl.it blocca download immagini (403) | Usato estrazione da PDF con PyMuPDF |
| PowerShell UTF-8 BOM | Usato `System.IO.File.WriteAllText` con `UTF8Encoding(false)` |
| Render Free cold start | ~30s dopo 15 min inattività. Upgrade a Starter (7 USD/mese) per sempre attivo |
| Pagine legali attirano query generiche | Privacy policy e cookie sono vicine a qualunque domanda che nomini "il sito". Risolto con la soglia di rilevanza dopo il reranking: i chunk sotto 4/10 non entrano nel contesto |
| Il reranker vedeva pochi candidati | Pool portato a 20: chunk quasi identici dello stesso documento riempivano tutti gli slot ed escludevano la pagina pertinente |
| Testo PDF su due colonne interfogliato | `extract_text()` alterna le righe delle due colonne. Il modello se la cava, ma è il motivo per cui alcune schede risultano confuse a leggersi |
| Righe di tabella fuse da pdfplumber | Corretto: dove mancano i righelli di separazione (ARGO.pdf p.5, XLC_PILOTS.pdf p.14) la riga viene divisa solo se **ogni** cella piena contiene lo stesso numero di valori, e un intervallo (`2-20`, `50-65`) conta come valore unico. Su 226 righe candidate del corpus ne tocca 2, entrambe realmente fuse |
| Programmi di calcolo non citabili | Stanno nell'area riservata: il crawler anonimo non li vede. Recuperati dal sitemap e indicizzati come puntatori con la nota che serve il login |

---

## 9-bis. Difetti noti ancora aperti

- **`"Quanto pesa la XLC 400 DN 300?"`** nella formulazione secca risponde di non
  avere il dato, che invece esiste (405 kg). Con "…e le sue quote A e B" funziona.
  La pagina con la tabella non entra fra i candidati: qui "300" è una taglia e
  "400" è la serie, e la domanda in prosa somiglia poco a una riga di numeri.
  È un rifiuto, non un dato sbagliato.
- **`ANALYTICS_TOKEN` non è impostato su Render**: finché manca,
  `/api/analytics/*` e `/api/feedback/stats` restano leggibili da chiunque
  conosca il path. L'app lo segnala nei log all'avvio.
- **Righe di tabella fuse da pdfplumber** dove il PDF non ha righelli di
  separazione: il dato resta etichettato (`A mm = 80 110`) ma la coppia
  taglia→valore è ambigua. Riguarda 2 righe su 226 nel corpus.
- Le modifiche in `avatar-poc/` (config puntato a localhost) non sono committate.

---

## 10. Prossimi Passi Possibili

- Integrare widget nel sito csasrl.it (aggiungere script tag nel tema WordPress)
- Upgrade Render a Starter per eliminare cold start
- Aggiungere catalogo generale PDF (non solo schede tecniche)
- Implementare feedback utente (thumbs up/down)
- Analytics: tracciare domande più frequenti
