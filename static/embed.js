/*
 * static/embed.js — incorpora il chatbot CSA in un sito esterno (csasrl.it).
 *
 * Uso, una riga sola nel sito:
 *   <script src="https://csa-chatbot.onrender.com/static/embed.js" defer></script>
 *
 * PERCHE' UN IFRAME E NON IL WIDGET DIRETTO
 * Il widget usa URL relativi (`/api/chat`): incollato nel tema chiamerebbe
 * csasrl.it/api/chat, che non esiste. Dentro l'iframe le chiamate restano
 * sull'origine di Render, quindi funzionano e non toccano nemmeno la CORS.
 * In piu' il CSS del tema (WordPress + Elementor definiscono regole globali su
 * button, div, input) non puo' entrare, e quello del widget non puo' uscire.
 *
 * L'iframe cambia misura: chiuso e' grande quanto il pulsante, altrimenti un
 * rettangolo trasparente coprirebbe l'angolo del sito intercettando i clic.
 */
(function () {
  "use strict";

  if (window.__csaChatbotCaricato) return;   // due <script> per errore: uno solo vale
  window.__csaChatbotCaricato = true;

  var script = document.currentScript;
  // L'origine da cui questo file e' stato servito: cosi' l'indirizzo dell'API
  // non va scritto a mano e non puo' divergere.
  var ORIGINE = script
    ? new URL(script.src, location.href).origin
    : "https://csa-chatbot.onrender.com";

  // Quanto stare sollevati dal fondo. Serve dove l'angolo e' gia' occupato:
  // su csasrl.it c'e' un pulsante WhatsApp, e la bolla ci finiva sopra
  // rendendo scomodi tutti e due. Si imposta sul tag script:
  //   <script src="..." data-bottom="76" defer></script>
  // Dove l'angolo e' gia' occupato — su csasrl.it c'e' un pulsante WhatsApp —
  // il nostro si sposta per non finirgli sopra. Due modi, sul tag script:
  //   data-right="190"  lo mette DI FIANCO (spostato da destra)
  //   data-bottom="76"  lo mette SOPRA (sollevato dal fondo)
  var SCOSTAMENTO_DX = Math.max(0, parseInt((script && script.dataset.right) || "0", 10) || 0);
  var SCOSTAMENTO_BASSO = Math.max(0, parseInt((script && script.dataset.bottom) || "0", 10) || 0);

  // Su uno schermo stretto due pastiglie in fila non ci stanno: la nostra
  // torna sopra l'altra invece di uscire dallo schermo.
  var LARGHEZZA_MINIMA_AFFIANCO = 600;

  function affiancabile() {
    return (window.innerWidth || 0) >= LARGHEZZA_MINIMA_AFFIANCO;
  }

  function posizione() {
    if (SCOSTAMENTO_DX && affiancabile()) {
      return { destra: SCOSTAMENTO_DX, basso: 0 };
    }
    // Senza spazio di fianco: sopra, con l'alzata dichiarata o quanto basta
    // a scavalcare un pulsante d'angolo tipico.
    return { destra: 0, basso: SCOSTAMENTO_BASSO || (SCOSTAMENTO_DX ? 76 : 0) };
  }

  var ALZATA = posizione().basso;

  var LATO_CHIUSO = { larghezza: "96px", altezza: "96px" };  // sostituita dalla misura vera
  var aperto = false;
  var LATO_APERTO = { larghezza: "400px", altezza: "640px" };

  // La lingua della pagina che ospita il widget. Da dentro l'iframe, che sta
  // su un'altra origine, <html lang> non e' leggibile: va passata di qua.
  // Su csasrl.it/en una scritta italiana nel riquadro stonava, mentre le
  // risposte erano gia' nella lingua giusta.
  function linguaPagina() {
    var lang = document.documentElement.getAttribute("lang")
      || (document.querySelector('meta[property="og:locale"]') || {}).content
      || navigator.language
      || "";
    return String(lang).toLowerCase().slice(0, 2);
  }

  var iframe = document.createElement("iframe");
  var lingua = linguaPagina();
  // Su uno schermo stretto la pastiglia con la scritta non convive con gli
  // altri pulsanti d'angolo: resta la sola icona. Lo decide qui perche' dentro
  // l'iframe la larghezza dello schermo non e' quella vera.
  // Versione del widget. embed.js scade in 5 minuti, quindi entro 5 minuti
  // tutti ricevono questo file; cambiando qui il numero, la pagina dentro
  // l'iframe diventa un URL nuovo e nessuna copia vecchia puo' sopravvivere.
  // Senza, un browser che aveva gia' visto il widget continuava a mostrare il
  // pulsante precedente anche a correzione pubblicata.
  var VERSIONE = "5";

  var parametri = ["v=" + VERSIONE];
  if (lingua) parametri.push("lang=" + encodeURIComponent(lingua));
  if ((window.innerWidth || 9999) < 480) parametri.push("compact=1");
  iframe.src = ORIGINE + "/" + (parametri.length ? "?" + parametri.join("&") : "");
  iframe.title = "Assistente CSA";
  iframe.setAttribute("allowtransparency", "true");
  iframe.style.cssText = [
    "position:fixed",
    "bottom:" + posizione().basso + "px",
    "right:" + posizione().destra + "px",
    "width:" + LATO_CHIUSO.larghezza,
    "height:" + LATO_CHIUSO.altezza,
    "border:0",
    "background:transparent",
    "z-index:2147483000",       // sopra i popup di Elementor, sotto i massimi
    "color-scheme:normal",
    "transition:width .18s ease, height .18s ease"
  ].join(";");

  function applica(misura) {
    iframe.style.width = misura.larghezza;
    iframe.style.height = misura.altezza;
  }

  // Mai piu' grande dello schermo. Un breakpoint ("se sotto i 480px allora
  // telefono") indovina, e indovina male sui browser che riportano una
  // viewport logica diversa da quella reale: il riquadro finiva mezzo fuori
  // dal bordo. Il minimo fra la misura voluta e lo spazio disponibile e'
  // giusto su qualunque schermo senza doverlo classificare.
  function misuraApertura() {
    var voluteL = parseInt(LATO_APERTO.larghezza, 10);
    var voluteA = parseInt(LATO_APERTO.altezza, 10);
    // Lo spazio a sinistra dello scostamento: un riquadro largo quanto lo
    // schermo, ma spostato dal bordo destro, uscirebbe da quello sinistro.
    var scostamento = posizione().destra;
    var maxL = window.innerWidth ? Math.max(0, window.innerWidth - scostamento) : 0;
    // L'alzata mangia altezza utile: senza sottrarla il riquadro aperto
    // sporgerebbe oltre il bordo alto dello schermo.
    var maxA = window.innerHeight ? Math.max(0, window.innerHeight - ALZATA) : 0;
    // Una viewport di 0 non e' uno schermo minuscolo, e' un browser che non
    // sa ancora rispondere (scheda in secondo piano, misura durante il
    // caricamento): restringere a 0 renderebbe il riquadro invisibile.
    return {
      larghezza: (maxL > 0 ? Math.min(voluteL, maxL) : voluteL) + "px",
      altezza: (maxA > 0 ? Math.min(voluteA, maxA) : voluteA) + "px"
    };
  }

  // Se lo schermo cambia (rotazione del telefono, finestra ridimensionata)
  // mentre il riquadro e' aperto, va rimisurato: restava della misura di prima.
  function collocaAiBordi() {
    var p = posizione();
    ALZATA = p.basso;
    iframe.style.right = p.destra + "px";
    iframe.style.bottom = p.basso + "px";
  }

  window.addEventListener("resize", function () {
    collocaAiBordi();
    applica(aperto ? misuraApertura() : LATO_CHIUSO);
  });

  window.addEventListener("message", function (evento) {
    if (evento.origin !== ORIGINE) return;              // solo dal nostro iframe
    var dati = evento.data;
    if (!dati || dati.source !== "csa-chatbot") return;
    if (dati.type === "size") {
      // Il widget dice quanto misura il suo pulsante: l'etichetta cambia
      // lunghezza da una lingua all'altra, e un iframe piu' largo del
      // pulsante intercetterebbe i clic sulla pagina sotto.
      LATO_CHIUSO = {
        larghezza: Math.max(48, dati.larghezza | 0) + "px",
        altezza: Math.max(48, dati.altezza | 0) + "px"
      };
      if (!aperto) applica(LATO_CHIUSO);
      return;
    }
    if (dati.type === "open") {
      aperto = true;
      // Il riquadro resta ancorato al PROPRIO pulsante, non all'angolo:
      // riportarlo nell'angolo lo faceva finire sopra il pulsante WhatsApp,
      // che e' proprio quello che lo scostamento doveva evitare. Allineato al
      // suo pulsante e' anche dove uno se lo aspetta, avendo cliccato li'.
      collocaAiBordi();
      applica(misuraApertura());
    } else if (dati.type === "close") {
      aperto = false;
      collocaAiBordi();
      applica(LATO_CHIUSO);
    }
  });

  function inserisci() {
    document.body.appendChild(iframe);
  }

  if (document.body) inserisci();
  else document.addEventListener("DOMContentLoaded", inserisci);
})();
