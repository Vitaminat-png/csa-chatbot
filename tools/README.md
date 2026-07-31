# tools/ — verifica del chatbot

Due strumenti per controllare le risposte dopo una modifica, senza avviare il server.
Richiedono `.env` con le chiavi (leggono Pinecone e OpenAI davvero).

## probe.py — interroga il bot e mostra tutto

```bash
python tools/probe.py "A che pressione lavora la ITALICA 353?"
python tools/probe.py "d1" "d2" "d3"                      # in parallelo
python tools/probe.py --chat "prima domanda" "e basta?"   # turni con memoria
python tools/probe.py --context "domanda"                 # stampa il contesto completo
python tools/probe.py --json "domanda"                    # output JSON
```

Per ogni domanda stampa la lingua rilevata, le fonti recuperate con le applicazioni
documentate, la risposta, e per ogni link se compare nel contesto e se è raggiungibile.
Applica la stessa validazione dei link che l'API applica in produzione, quindi mostra
quello che vedrebbe davvero un utente.

`--context` è lo strumento da usare quando una risposta è sbagliata: quasi sempre la
domanda è "il dato era nel contesto?", e la risposta cambia completamente la diagnosi.

## verify_answers.py — regressione sui dati tecnici

```bash
python tools/verify_answers.py
```

Esegue una batteria di domande con i valori attesi verificati contro le schede
tecniche, più i casi di rilevamento lingua. Da rieseguire dopo ogni modifica a
`api/retrieval.py`, `api/prompt.py` o `api/model_index.py`.

I casi non sono arbitrari: ciascuno è un difetto realmente trovato e corretto.
Se uno torna rosso, è una regressione, non un test fragile.

## Cosa NON coprono

`pytest` copre la logica pura (codici modello, lingua, link, tabelle) e non consuma
API. Questi due strumenti coprono il comportamento end-to-end, che dipende dal
modello e varia leggermente fra esecuzioni: se un caso fallisce una volta su cinque
è instabilità, non necessariamente un difetto.
