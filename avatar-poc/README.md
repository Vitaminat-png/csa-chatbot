# CSA Avatar Real-Time POC

Proof-of-concept di avatar femminile fotorealistico collegato al backend CSA con pipeline real-time basata su **Simli WebRTC**.

## Obiettivo di questa versione

Il vecchio POC D-ID generava un video asincrono con latenze dell’ordine di 8-20 secondi.  
Questa versione elimina completamente D-ID e passa a una pipeline ottimizzata per latenza bassa:

1. l’utente scrive una domanda
2. il frontend apre o riusa una sessione Simli WebRTC
3. il frontend chiama `POST /api/chat/stream` sul backend CSA
4. i token SSE vengono accumulati in frasi brevi
5. ogni frase viene inviata a `POST /api/avatar/tts`
6. il backend genera audio PCM streaming con OpenAI TTS e lo ricampiona a 16 kHz
7. il frontend invia i chunk PCM a Simli via WebSocket
8. Simli anima l’avatar in tempo reale con lip sync live

Risultato atteso: avvio della voce entro circa 2-3 secondi percepiti, senza aspettare la risposta completa.

## Scelta tecnica

### Provider avatar

Scelta primaria: **SimliAI**

Motivi:

- WebRTC real-time
- free tier dichiarato di 50 minuti/mese
- latenza nettamente più adatta a una chat rispetto al vecchio flusso D-ID
- architettura compatibile con audio PCM progressivo

### TTS

Scelta primaria: **OpenAI TTS**

Motivi:

- `OPENAI_API_KEY` è già disponibile nel progetto
- supporta streaming server-side
- consente di controllare direttamente chunking, latenza e testo
- evita di delegare a un motore TTS esterno poco controllabile nel POC

## Architettura implementata

### Frontend

File: `avatar-poc/index.html`

- UI rifatta da zero
- 2 avatar selezionabili
- 2 voci femminili selezionabili
- sessione Simli WebRTC diretta dal browser
- lettura streaming da `POST https://csa-chatbot.onrender.com/api/chat/stream`
- chunking progressivo del testo in frasi
- invio realtime dei chunk audio verso Simli

### Config

File: `avatar-poc/config.js`

Contiene:

- URL backend CSA remoto
- URL backend locale avatar
- 2 avatar selezionabili
- 2 voci femminili
- parametri sessione Simli
- soglie flush TTS

### Backend

File: `api/main.py`

Nuovi endpoint:

- `POST /api/avatar/session`
  - crea session token Simli
  - recupera ICE servers temporanei
  - restituisce token + websocket URL al browser

- `POST /api/avatar/tts`
  - usa OpenAI TTS in streaming
  - produce audio PCM
  - ricampiona 24 kHz -> 16 kHz per Simli
  - streamma i byte al frontend

File: `api/models.py`

- rimossi i modelli D-ID
- aggiunti i modelli per sessione Simli, ICE e TTS

## Setup rapido

### 1. Recupera la Simli API key

Dashboard:

- `https://app.simli.com/`

Nel file `.env` aggiungi:

```env
SIMLI_API_KEY=YOUR_SIMLI_API_KEY_HERE
OPENAI_API_KEY=sk-...
```

### 2. Crea almeno 2 face ID in Simli

Nel dashboard Simli crea o seleziona due avatar fotorealistici e copia i relativi `faceId`.

Apri `avatar-poc/config.js` e sostituisci:

- `REPLACE_WITH_SIMLI_FACE_ID_1`
- `REPLACE_WITH_SIMLI_FACE_ID_2`

con i valori reali.

Questo passaggio è obbligatorio. Il frontend mostra esplicitamente “config incompleta” finché i placeholder restano invariati.

### 3. Avvia il backend locale

```bash
uvicorn api.main:app --reload
```

### 4. Avvia un web server statico locale

Esempio:

```bash
python -m http.server 5501
```

Poi apri:

```text
http://127.0.0.1:5501/avatar-poc/
```

## Come usare la demo

1. scegli uno dei 2 volti
2. scegli una delle 2 voci
3. clicca `Connetti avatar` oppure invia direttamente una domanda
4. scrivi una domanda sul catalogo CSA
5. clicca `Invia e parla subito`
6. osserva il testo che arriva in streaming e l’avatar che parte prima della fine della risposta

## Configurazioni importanti

### `avatar-poc/config.js`

Campi principali:

- `chatbotApiBase`
- `avatarBackendBase`
- `faces[].simliFaceId`
- `voices[].id`
- `simliSession.maxSessionLength`
- `simliSession.maxIdleTime`
- `tts.streamChunkBytes`
- `tts.sentenceFlushChars`

### `env.example`

Ora espone:

- `SIMLI_API_KEY`
- `OPENAI_API_KEY`

### `render.yaml`

La variabile deploy per l’avatar POC è stata aggiornata da `D_ID_API_KEY` a `SIMLI_API_KEY`.

## Cosa è stato eliminato

Rimosso completamente il flusso D-ID:

- nessun `POST /api/avatar/respond`
- nessun polling `GET /api/avatar/status/{talk_id}`
- nessun render video asincrono
- nessuna dipendenza dal vecchio provider

## Limiti residui

- servono due `faceId` reali Simli impostati manualmente
- la latenza totale dipende anche dal tempo di risposta del backend CSA remoto
- OpenAI TTS non garantisce accento italiano perfettamente nativo
- il chunking del testo è euristico e ottimizzato per demo, non ancora production-grade

## Acceptance criteria: stato attuale

- avatar realtime: **implementato architetturalmente**
- lip sync realtime: **implementato via Simli WebRTC**
- voce femminile: **implementata via OpenAI TTS**
- almeno 2 avatar: **configurati**
- collegamento backend CSA reale: **implementato**
- browser only: **sì**

Nota importante: per una demo reale servono i due `faceId` Simli validi. Senza quelli il POC non può completare la connessione WebRTC al provider.

## Alternative se Simli non va bene

Fallback consigliati:

1. **Tavus**
2. **HeyGen Streaming Avatar**

Entrambe sono alternative più coerenti con il requisito realtime rispetto al vecchio approccio D-ID.