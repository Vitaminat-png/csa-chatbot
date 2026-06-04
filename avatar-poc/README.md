# CSA Avatar Bot POC

Proof-of-concept di avatar bot femminile fotorealistico collegato al backend CSA esistente.

## Scelta tecnica consigliata

Per questo POC ho scelto **D-ID** come opzione primaria.

### Perché D-ID

- API semplice per creare un talking-head da una singola foto
- supporta testo -> voce -> video in un'unica pipeline
- trial gratuito disponibile, anche se con watermark
- integrazione web molto più semplice di SadTalker/MuseTalk
- più adatto a una demo rapida rispetto a LiveKit + stack avatar custom

### Opzioni valutate

#### 1. D-ID
- **Pro:** molto semplice per POC, qualità visiva buona, talking photo diretto, integrazione REST pulita
- **Contro:** non è realtime puro, il render spesso richiede più di 5 secondi, watermark nel trial
- **Verdetto:** miglior compromesso per demo dimostrabile in poco tempo

#### 2. HeyGen
- **Pro:** qualità alta, buon catalogo avatar/voice, API moderne
- **Contro:** più costoso per video generato, latenza tipicamente da minuti e non da secondi, meno adatto a flusso conversazionale rapido
- **Verdetto:** ottimo per marketing video, meno ideale per questa demo chatbot

#### 3. SimliAI
- **Pro:** realtime, free tier interessante, latenza potenzialmente migliore
- **Contro:** integrazione più complessa, focus WebRTC/streaming, flusso audio realtime più impegnativo, selezione avatar meno immediata per una pagina standalone
- **Verdetto:** migliore opzione per una V2 realtime, non la più semplice per questo POC livello 2

#### 4. SadTalker / MuseTalk
- **Pro:** open source, massimo controllo, nessun costo SaaS diretto
- **Contro:** richiede GPU e setup pesante; SadTalker spesso troppo lento per demo; MuseTalk più promettente ma molto più complesso da far girare bene
- **Verdetto:** scartato per tempo/setup/costi infrastruttura

#### 5. LiveKit + avatar open-source
- **Pro:** architettura molto flessibile e realtime
- **Contro:** troppa complessità per questo obiettivo, più componenti da orchestrare
- **Verdetto:** overkill per un POC rapido

## Architettura del POC

Flusso:

1. utente scrive una domanda in `avatar-poc/index.html`
2. il frontend chiama `POST https://csa-chatbot.onrender.com/api/chat` per ottenere la risposta CSA
3. il frontend chiama il backend locale `POST /api/avatar/respond`
4. il backend locale inoltra il testo a D-ID usando la foto del volto selezionato e la voce scelta
5. il frontend esegue polling su `GET /api/avatar/status/{talk_id}`
6. quando il video è pronto, viene mostrato nel player

## File creati

- `avatar-poc/index.html` — UI standalone del POC
- `avatar-poc/config.js` — configurazione frontend
- `avatar-poc/README.md` — setup e valutazione opzioni
- `api/main.py` — proxy backend per D-ID
- `api/models.py` — schemi request/response avatar

## Requisiti

- Python environment del progetto già funzionante
- una API key D-ID
- backend FastAPI del progetto avviato in locale

## Setup rapido

### 1. Aggiungi la key D-ID

Nel tuo `.env`:

```env
D_ID_API_KEY=API_USERNAME:API_PASSWORD
```

Nota: D-ID usa Basic Auth. La key generata nello studio è nel formato `username:password`.

### 2. Avvia il backend locale

Esempio:

```bash
uvicorn api.main:app --reload
```

Di default il frontend del POC punta a:

- CSA chatbot remoto: `https://csa-chatbot.onrender.com`
- backend avatar locale: `http://127.0.0.1:8000`

Se vuoi cambiare URL, modifica `avatar-poc/config.js`.

### 3. Apri il frontend

Apri `avatar-poc/index.html` con Live Server o qualsiasi static server locale.

Per esempio:

```bash
python -m http.server 5501
```

Poi apri:

```text
http://127.0.0.1:5501/avatar-poc/
```

## Come usare la demo

- seleziona uno dei 3 volti femminili
- seleziona una delle 3 voci femminili
- scrivi una domanda sul catalogo CSA
- clicca `Genera risposta avatar`
- il testo della risposta compare subito
- il video avatar viene mostrato appena D-ID termina il render

## Limiti attuali del POC

- **latenza:** il target `<5s` non è sempre realistico con D-ID; per il trial aspettati più spesso `8-20s`
- **watermark:** probabile nel piano trial/free
- **non realtime:** è video generato asincrono, non stream realtime
- **volti demo:** il POC usa URL immagine pubblici come sorgente talking-head
- **voci:** il POC usa voci Microsoft gestite da D-ID, non ElevenLabs/OpenAI TTS separato

## Miglior upgrade successivo

Se vuoi avvicinarti davvero al requisito `<5s`, la strada migliore per la prossima iterazione è:

- passare a **SimliAI** per rendering realtime
- usare **OpenAI TTS o ElevenLabs** per voce
- opzionalmente aggiungere input microfono e STT

## Stima costi produzione

### D-ID
- trial: gratuito con limiti e watermark
- pricing API ufficiale: non facilmente estraibile via fetch statico, quindi va verificato direttamente sul sito D-ID prima della messa in produzione
- riferimento emerso dai risultati indicizzati: trial di 14 giorni con 3 minuti

### HeyGen API
- pay-as-you-go da **$5** minimi
- Photo Avatar IV/V circa **$0.05/sec**
- quindi ~**$3/minuto**
- buona qualità, ma costi più alti per chatbot frequente

### Simli
- **$10 di credito iniziale**
- **50 minuti/mese** nel free tier
- poi pay-as-you-go con sconti volume
- potenzialmente il candidato migliore per una versione conversazionale più fluida

### Open source self-hosted
- software gratis, ma costo GPU e manutenzione alto
- per MuseTalk/SadTalker in produzione il costo reale diventa infrastruttura + DevOps + stabilità

## Raccomandazione finale

Per una **demo rapida e dimostrabile oggi**, usa questo POC con **D-ID**.

Per una **V2 più fluida e vicina all'esperienza live**, valuta **SimliAI**.