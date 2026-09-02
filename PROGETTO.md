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
| Catalogo generale | `ingest/catalog_ingest.py` | `catalog` | Catalogo italiano con metadati per famiglia (2516 chunk) |
| XLC engineering (IT/EN/FR/ES) | `ingest/xlc_ingest.py` | default | Riferimento aggiornato su serie XLC 300/400 |
| Contenuto sito csasrl.it | `ingest/site_crawler.py` | default | Testo delle pagine del sito, in 4 lingue |

Il catalogo è escluso da `pdf_ingest` anche se sta in `docs/`: lo gestisce
`catalog_ingest.py`, che lo indicizza nel namespace `catalog` con i metadati di
sezione e famiglia. Processarlo in entrambi produceva una seconda copia peggiore
delle stesse 400+ pagine.

Gli script PDF condividono `ingest/pdf_extract.py`, che serializza le
tabelle e ripulisce il rumore dei grafici — vedi le decisioni qui sotto.
(`catalog_ingest.py` ha il suo formato tabella a pipe, ma dal 31/07/2026 la
pulizia della griglia — righe fuse comprese — passa dalla `_clean_table`
condivisa: prima no, e le righe fuse del catalogo restavano fuse.)

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
- **Il peso si chiede col verbo, non col sostantivo** (`_WEIGHT_WORDS` in
  `api/retrieval.py`). Il pin che tiene in contesto il chunk con la colonna del
  peso si attivava su `peso`/`pesi` ma non su `pesa`: `"Quanto pesa la XLC 400
  DN 300?"` non chiedeva nessuna colonna, la pagina 12 perdeva il tetto per
  documento contro sei pagine intitolate "Dati tecnici" e il bot rispondeva di
  non avere il dato mentre la riga diceva 405 kg. Il chunk giusto era già fra i
  candidati (0.648, `model_match`): non era un problema di recupero ma di
  riconoscimento della domanda. Coperte anche le forme inglesi, francesi e
  spagnole, verbo incluso.
- **Le etichette di colonna sono regex, perché il corpus non è uniforme**: le
  XLC engineering scrivono `Weight (Kg)`, le schede prodotto `Weight Kg` senza
  parentesi, la ATHENA abbrevia in `Wt Kg`, la VRCD toglie lo spazio
  (`Weight(Kg)`). Una lista di stringhe esatte spinnava silenziosamente la metà
  del corpus che non nominava: con la sola `"Weight Kg"` nessuna pagina XLC
  engineering era pinnabile, con la sola `"Weight (Kg)"` nessuna scheda. Le
  grafie vere sono state estratte dai PDF, non dedotte.
- **Un follow-up corto che nomina un prodotto non viene espanso**
  (`build_search_query`): è un cambio di argomento, non una continuazione.
  L'espansione col turno precedente serve ai messaggi senza soggetto ("e
  basta?"), ma il registry tiene solo il modello più specifico nominato: dopo
  una domanda sulla XLC 300, "athena che valvola è?" cercava "…XLC 300 DN 300
  athena…" dove `xlc 300` (2 token) batte `athena` (1) — e il bot rispondeva
  su ATHENA con documenti XLC, poi su "lynx?" con documenti ATHENA, rifiutando
  entrambe subito dopo aver risposto fluentemente.
- **Il pin segue il diritto di risposta, non l'ordine del pool**
  (`_order_holders`): fra i chunk che contengono la colonna richiesta vince la
  scheda del modello nominato, poi la sua famiglia, poi la serie nominata nel
  heading, e solo a parità l'ordine del pool. Due i difetti che ha chiuso:
  per "Quanto pesa la XLC 300 DN 300" veniva pinnata la tabella della serie
  400 (coseno un filo più alto) e il modello — rifiutandosi *giustamente* di
  leggerla per una 300 — negava un dato che stava a pagina 20; e per la FOX
  SUB il restore anteponeva due tabelle di catalogo con la stessa colonna, il
  pin le prendeva in ordine di pool, e il peso arrivava da un altro prodotto
  (74 kg invece di 44,5). Vale anche per il ripristino post-cap.
- **Fallback per lingua**: il rifiuto era una frase italiana che il modello
  doveva "tradurre se necessario", e spesso non lo faceva. Ora ogni lingua ha la
  sua, e per le lingue fuori tabella il modello scrive in quella dell'utente.
- **Lingua ereditata dalla conversazione**: un follow-up breve ("e basta?") non
  ha segnale e ricadeva sull'inglese, ribaltando la lingua a metà dialogo. Ora
  eredita dal turno precedente, e un pareggio non viene più risolto per ordine
  del dizionario (che premiava sempre l'italiano).
- **Nomi storpiati** (`_fuzzy_align` in `api/model_index.py`): "atena",
  "italika", "ciclops" rispondevano "non ho informazioni" con la scheda nel
  corpus — il match del registry era letterale. Un token sconosciuto viene
  allineato a un **nome di famiglia** solo se: ≥5 lettere (a 4, "solo" è a un
  edit da EOLO), stessa iniziale ("largo" non diventa ARGO), candidato unico.
  Bersagli solo le famiglie, non i pezzi di nome file ("sizing", "engineering").
- **Ponte famiglia→catalogo** (`find_family` + fetch su namespace catalog):
  le ITALICA non hanno tabella dimensioni in nessuna scheda — sta solo nella
  pagina di famiglia del catalogo (p423), che niente collegava al nome del
  modello. Su una domanda da tabella che nomina una famiglia, il retrieval
  interroga anche il catalogo filtrato per `product_family`. Gated sulla sonda
  tabelle per non allargare le domande ordinarie.
- **Il catalogo va etichettato a tre vie** (`_mark_family_catalogue_pages`):
  l'etichetta "DIFFERENT PRODUCT mai prendere i suoi numeri" applicata al
  catalogo vietava l'unica tabella ITALICA esistente (rifiuto); esentare il
  catalogo in blocco faceva rispondere i 26 kg della tabella FOX (p29, che il
  reranker metteva prima). Ora: pagina di catalogo della famiglia nominata →
  nota affermativa; di un'altra famiglia → vietata; scheda variante → vietata
  come prima.
- **"DN"/"PN" non sono serie** (`_series_designations`): "DN 100" nella
  domanda e "DN100" nel heading della tabella FOX combaciavano come "serie
  condivisa" e marcavano la tabella FOX come LA scheda di una domanda ITALICA.
  Taglie e classi di pressione sono escluse dalle designazioni di serie.
- **La nota di serie vale per quote e pesi, non per le pressioni**: CSA
  pubblica le dimensioni una volta per serie, ma la pressione di esercizio e'
  per modello — la serie standard regge 25 bar mentre XLC 353, 380/480 e
  310 ND si fermano a 16. Con la nota attiva su una domanda di pressione il bot
  rispondeva 25 bar per tutte e tre: un componente da 16 bar dato per 25.
- **La pagina del sito non e' la scheda**: entrambe possono riguardare
  esattamente il modello chiesto, ma csasrl.it elenca le classi PN in
  commercio ("Pressione: 10-16-25 bar") dove la scheda dichiara il limite di
  esercizio. La scheda precede la pagina, ed e' etichettata come autorevole
  sui numeri.
- **Sezioni dentro un PDF** (`api/section_map.json`, generato da
  `ingest/build_section_map.py`): APOLLO_RPC.pdf documenta l'Apollo RP alle
  pagine 8-9 e l'Apollo RPC alle 10-11; SCS_AS.pdf accoda a p5 il kit GOLIA
  SUB; XLC_PILOTS.pdf contiene otto pilota. L'etichetta "questa e' LA scheda
  del modello chiesto" era per file: la quota A della RPC tornava 682 mm (il
  valore della RP) e il peso della SCS-AS 7,0-88,3 kg (quelli del kit). La
  mappa da anche il vocabolario dei prodotti che il registry non conosce,
  perche' costruito dai nomi file: il regolatore CSFL vive dentro
  XLC_PILOTS.pdf e prima non era raggiungibile.
- **Serie per pagina** (`api/page_series.json`): il catalogo dedica pagine
  distinte alla XLC 300, 500 e 600, tutte "XLC" nei metadati, e il chunk-tabella
  e' spesso un muro di cifre che non nomina nulla. Trattarle come equivalenti
  faceva rispondere il DN 80 della XLC 330 con i 20 kg della 500. La serie sta
  nel corpo della pagina e viene estratta una volta per tutte.
- **Le serie 500/600 hanno documenti propri**: il manuale engineering copre
  solo 300 e 400, e senza `XLC_500_SIZING.pdf` fra i documenti di serie il peso
  di una XLC 600 tornava con i 34 kg della 300. I documenti di serie hanno
  anche una query dedicata: condividendo gli slot con la famiglia XLC li
  perdevano tutti.
- **Un token solo e' un modello solo se il file porta quel nome**: "VRCA" e
  "ATHENA" sono codici completi, "XLC" e' una famiglia di cinquanta documenti
  il cui file base e' il generico XLC_ENG.pdf — accettarlo come codice esatto
  faceva rispondere "quanto pesa la XLC 600" da quel file.
- **Nomi parlati**: CSA scrive le serie doppie ("XLC 365/465-MCP" e' la scheda
  XLC_365_MCP.pdf) e i titoli portano il denominatore che il nome file lascia
  cadere (SATURNO_RFP.pdf si intitola "SATURNO 3F - RFP"). Le misure in pollici
  con frazione — "1 1/4" — sono codificate come 114. Senza queste tolleranze il
  bot rifiutava valori documentati nella scheda stessa.
- **Didascalia sopra ogni tabella** (`ingest/pdf_extract.py`): una pagina puo'
  portare piu tabelle della stessa forma — p5 di ATHENA_114.pdf ne ha tre, e le
  righe dell'ultima non hanno nemmeno la colonna DN. Serializzate senza
  didascalia erano indistinguibili, e la portata del modello piccolo veniva
  letta da un numero del testo speculare della barra laterale. Una didascalia
  sbagliata sarebbe peggio di nessuna, quindi il filtro scarta code di
  paragrafo, righe di assi e artefatti a lettere doppie.
- **Un identificativo per blocco, non per pagina** (`ingest/catalog_ingest.py`):
  il contatore dei pezzi ripartiva da zero a ogni blocco della pagina, quindi
  due tabelle sulla stessa pagina finivano sullo stesso id e la seconda
  cancellava la prima. Su questo catalogo erano **412 pezzi su 2516 (il 16%),
  sparsi su 158 pagine** — e l'ingestione dichiarava comunque 2516 vettori
  scritti, per questo non si era mai visto. Colpiva proprio le pagine con piu
  tabelle, cioe' quelle con i dati tecnici. Trovato il 02/09/2026 mentre si
  ricalcolavano gli id per ripulire i vettori superati.
- **Pagine prodotto del sito per categoria** (`api/site_products.json`,
  generato da `ingest/build_site_products.py`): l'elenco che il bot produce a
  "quali idranti fate?" nasce dall'indice delle schede tecniche, quindi un
  prodotto che il sito pubblica ma nessuna scheda indicizzata documenta non
  compariva — l'APOLLO RPC SMART rispondeva quando lo si nominava, ma spariva
  dagli elenchi. Gli slug delle categorie sono registrati nelle quattro lingue
  ("idranti", "pillar-fire-hydrants", "poteaux-incendie", "hidrante-de-columna")
  e basta la parola-chiave finale, purche' non generica: "valvole" da sola non
  tira dentro quindici pagine.
  Ci sono voluti tre passaggi, e il primo non bastava: la pagina entrava in
  contesto 3 volte su 3 ma il modello la citava 1 volta su 3, perche' costruiva
  l'elenco dalle schede e si fermava. Come per le applicazioni documentate, si
  e' risolto dicendolo come **fatto su quella fonte** ("prodotto della categoria
  chiesta, includilo nell'elenco") invece che come regola generale. Infine il
  tetto per fonte: con quello largo la scheda dell'Apollo RPC prendeva quattro
  slot e il terzo idrante non entrava.
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
- **Sagoma quotata insieme alle dimensioni** (`ingest/render_dimension_pages.py`
  → `static/products/dimensions/*.png` + `api/dimension_drawings.json`): una
  risposta che elenca "A = 230 mm, B = 82,5 mm" senza il disegno è una lista di
  numeri ciechi. La pagina della scheda che porta la tabella porta anche la
  sagoma con le lettere: viene resa in PNG e allegata via canale `images` del
  widget quando quella pagina è fra le fonti. Se la domanda nomina un modello,
  solo la sua scheda può contribuire (il retrieval riempie il contesto con
  pagine dimensioni di altri prodotti); le XLC usano un disegno per serie
  (p12→400, p20→300), uguale nelle 4 edizioni. Le PNG vanno committate:
  `docs/` non entra nell'immagine Docker. Rieseguire lo script quando si
  aggiunge una scheda.
- Le foto prodotto self-hosted senza file reale vengono saltate: la mappa
  segnaposto generava un 404 nascosto a ogni risposta (e 4 delle 8 "famiglie"
  mappate — dedalo, vortice, orbis, isis — non sono prodotti CSA).
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
│   └── test_20_questions.py  (98 test, 92 unit + 6 integration)
├── tools/
│   ├── probe.py            (interroga il bot: fonti, contesto, link verificati)
│   └── verify_answers.py   (regressione sui dati tecnici, end-to-end)
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
4. Rigenerare le sagome quotate (e committare le PNG nuove):
   ```bash
   python -m ingest.render_dimension_pages
   ```

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
| Righe di tabella fuse da pdfplumber | Corretto due volte. Prima versione: split solo nelle tabelle ≤3 righe (ARGO.pdf p.5, XLC_PILOTS.pdf p.14), se **ogni** cella piena contiene lo stesso numero di valori; un intervallo (`2-20`) conta come valore unico. Ma le tabelle CYCLOPS/GOLIA hanno 10+ righe con le righe centrali fuse (`Flanged 100 Flanged 150R` → `Weight Kg = 21,5 34`) e la guardia le saltava: la 150R pesava 57 kg (valore della 150 liscia). Seconda versione: la guardia sulla dimensione è sostituita da "riga interamente piena" — la firma vera di una riga fusa; le righe di continuazione (APOLLO) e i detriti grafici hanno celle vuote e restano intoccati. Censimento su 7637 righe dati: 21 candidate, tutte verificate a mano. I suffissi `R`/`*` sopravvivono allo split (150R ≠ 150) |
| Programmi di calcolo non citabili | Stanno nell'area riservata: il crawler anonimo non li vede. Recuperati dal sitemap e indicizzati come puntatori con la nota che serve il login |

---

## 9-bis. Difetti noti ancora aperti

- **Quota della GOLIA 3F 150R instabile (~1 volta su 3)**: la tabella ha
  "Flanged 150R" (A 235, 27,0 kg) e "Flanged 150" (A 300, 45,0 kg) su righe
  adiacenti, e il modello prende il peso dalla riga giusta ma ogni tanto la
  quota da quella sotto. Misurato il 02/09/2026: 2 esecuzioni su 3 corrette.
  Non e' un problema di recupero — entrambe le righe sono in contesto,
  correttamente separate — ma di lettura.
- **"Che idranti trovo sul vostro sito?" non elenca l'APOLLO RPC SMART**
  (0 su 3), mentre "quali modelli di idranti produce CSA?" e la versione
  inglese lo elencano 3 su 3. La parola "sito" tira in contesto pagine
  generiche che diluiscono le pagine prodotto.

- **Indice del sito non aggiornato**: la pagina categoria Idranti elenca tre
  prodotti (Apollo RP, Apollo RPC, APOLLO RPC SMART) e il bot ne cita due —
  lo SMART non e' nell'indice. Unico difetto dell'audit del 30/08/2026 ancora
  aperto: si chiude con `python -m ingest.site_crawler`, che pero' riscrive i
  vettori del sito nell'indice condiviso con la produzione.
- ~~Didascalie solo su ATHENA_114.pdf~~ — **chiuso il 02/09/2026**: reingerito
  l'intero corpus (115 schede, XLC engineering, catalogo) con le didascalie.
- **Aggiornamento documenti del 02/09/2026**: installate 12 schede aggiornate a
  giugno (SCF e varianti, VRCD_XN, XLC 325/340/350/510), il catalogo `v2` e la
  scheda GEMINA. Il Kv della Gemina DN 150 e' passato da 273 a **330**: i
  documenti "2026" italiano e inglese concordano, catalogo e scheda sono stati
  allineati **insieme** (aggiornarne uno solo avrebbe lasciato entrambi i valori
  nel corpus). Il corpus precedente e' in `docs/_backup_2026-09-02`.

- **`"CSA è certificata secondo la UNI EN 558?"` è il caso instabile della
  batteria**: su tre esecuzioni consecutive è fallito due volte e passato una,
  e nelle esecuzioni fallite il reranker era andato in timeout (18 s) — sul
  timeout il retrieval ricade sull'ordine Pinecone e salta la soglia di
  rilevanza. Da solo risponde correttamente. Non è il caso di considerarlo
  verde: è il segnale che `RERANK_TIMEOUT` è stretto quando l'API è lenta.
- **`ANALYTICS_TOKEN` non è impostato su Render**: finché manca,
  `/api/analytics/*` e `/api/feedback/stats` restano leggibili da chiunque
  conosca il path. L'app lo segnala nei log all'avvio.
- ~~Righe di tabella fuse da pdfplumber~~ **risolto il 31/07/2026**: la guardia
  "solo tabelle ≤3 righe" saltava le tabelle CYCLOPS/GOLIA (10+ righe con le
  centrali fuse) e il catalogo non passava affatto dallo splitter. Vedi §9.
  Re-ingest fatto per i 6 file toccati e per il catalogo.
- Le modifiche in `avatar-poc/` (config puntato a localhost) non sono committate.
- **`static/products/` è vuota** (solo README): le foto prodotto del widget non
  compaiono mai, né in locale né sul live — il widget nasconde i 404, quindi
  nessun errore visibile. csasrl.it blocca il download automatico (403): le
  immagini vanno caricate a mano o si passa agli URL wp-content diretti.

---

## 10. Prossimi Passi Possibili

- Integrare widget nel sito csasrl.it (aggiungere script tag nel tema WordPress)
- Upgrade Render a Starter per eliminare cold start
- Aggiungere catalogo generale PDF (non solo schede tecniche)
- Implementare feedback utente (thumbs up/down)
- Analytics: tracciare domande più frequenti
