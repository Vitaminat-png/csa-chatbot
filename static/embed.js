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

  var LATO_CHIUSO = { larghezza: "96px", altezza: "96px" };
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
  iframe.src = ORIGINE + "/" + (lingua ? "?lang=" + encodeURIComponent(lingua) : "");
  iframe.title = "Assistente CSA";
  iframe.setAttribute("allowtransparency", "true");
  iframe.style.cssText = [
    "position:fixed",
    "bottom:0",
    "right:0",
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
    var maxL = window.innerWidth || 0;
    var maxA = window.innerHeight || 0;
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
  window.addEventListener("resize", function () {
    if (iframe.style.width !== LATO_CHIUSO.larghezza) applica(misuraApertura());
  });

  window.addEventListener("message", function (evento) {
    if (evento.origin !== ORIGINE) return;              // solo dal nostro iframe
    var dati = evento.data;
    if (!dati || dati.source !== "csa-chatbot") return;
    if (dati.type === "open") applica(misuraApertura());
    else if (dati.type === "close") applica(LATO_CHIUSO);
  });

  function inserisci() {
    document.body.appendChild(iframe);
  }

  if (document.body) inserisci();
  else document.addEventListener("DOMContentLoaded", inserisci);
})();
