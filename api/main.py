"""
api/main.py
-----------
FastAPI application exposing the CSA AI Chatbot API.

Endpoints:
  POST /api/chat          — returns ChatResponse (JSON)
  POST /api/chat/stream   — Server-Sent Events streaming response
  GET  /health            — health check

CORS is configured to allow csasrl.it and localhost (dev).
"""

from __future__ import annotations

import audioop
import json
import logging
import os
import pathlib
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from api.analytics import log_query, router as analytics_router
from api.feedback import router as feedback_router
from api.links import StreamingLinkSanitizer, sanitize_links
from api.models import (
    AvatarSessionRequest,
    AvatarSessionResponse,
    AvatarTTSRequest,
    ChatRequest,
    ChatResponse,
    ProductImage,
    SimliIceServer,
    Source,
)
from api.product_images import get_dimension_drawings, get_images_for_families
from api.prompt import build_system_prompt
from api.model_index import find_model_sources
from api.retrieval import build_context_string, build_search_query, retrieve

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CSA AI Chatbot API",
    version="1.0.0",
    description="RAG-powered chatbot for CSA S.r.l. industrial valves.",
)

ALLOWED_ORIGINS = [
    "https://csasrl.it",
    "https://www.csasrl.it",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:8080",
    "http://127.0.0.1:5500",   # VS Code Live Server default
    "http://127.0.0.1:5501",
    "http://127.0.0.1:8080",
    "null",                     # file:// origin for local widget testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(feedback_router)
app.include_router(analytics_router)

# Serve product images from /static/products/
_STATIC_PATH = pathlib.Path(__file__).parent.parent / "static"
if _STATIC_PATH.exists():
    class _StaticConScadenzaBreve(StaticFiles):
        """
        File statici con una scadenza corta ed esplicita.

        embed.js sta nel sito del cliente: senza Cache-Control il browser si
        inventa la scadenza e una correzione poteva metterci ore ad arrivare.
        Cinque minuti tengono bassa la latenza degli aggiornamenti senza
        rinunciare alla cache; le immagini delle sagome, che non cambiano mai,
        pagano pochissimo.
        """

        def file_response(self, *args, **kwargs):
            risposta = super().file_response(*args, **kwargs)
            risposta.headers["Cache-Control"] = "public, max-age=300"
            return risposta

    app.mount("/static", _StaticConScadenzaBreve(directory=str(_STATIC_PATH)), name="static")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SIMLI_API_KEY = os.environ.get("SIMLI_API_KEY", "")
CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
MAX_TOKENS = 1024
MAX_HISTORY = 10  # max conversation turns to include for context

async_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
SIMLI_API_BASE = "https://api.simli.ai"
SIMLI_P2P_WEBSOCKET_URL = "wss://api.simli.ai/compose/webrtc/p2p"


def _build_messages(system_prompt: str, history, current_message: str) -> list[dict]:
    """Build the messages list for GPT, prepending conversation history."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for h in (history or [])[-MAX_HISTORY:]:
        role = h.role if h.role in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": h.content})
    msgs.append({"role": "user", "content": current_message})
    return msgs


def _allowed_links(sources: list[Source]) -> list[str]:
    """
    URLs the answer is permitted to cite — those actually retrieved.

    Includes each page's other language editions: they come from the same
    verified metadata, and without them a legitimate "and in English?" answer
    had its correct URL stripped by the link check.
    """
    allowed: list[str] = []
    for source in sources:
        if source.url:
            allowed.append(source.url)
        allowed.extend(source.url_alternates.values())
    return allowed


# What the visitor sees when OpenAI or Pinecone is unavailable, in each language
# the site serves. Previously nothing wrapped those calls, so a rate limit — the
# 200k tokens/minute ceiling is reached by a modest burst of visitors — or a
# provider timeout surfaced as a bare HTTP 500.
_UPSTREAM_ERROR_MESSAGE = {
    "it": "Il servizio è momentaneamente sovraccarico. Riprova tra qualche istante, "
          "oppure scrivi a info@csasrl.it.",
    "en": "The service is momentarily overloaded. Please try again in a few moments, "
          "or write to info@csasrl.it.",
    "fr": "Le service est momentanément surchargé. Merci de réessayer dans quelques "
          "instants, ou écrivez à info@csasrl.it.",
    "es": "El servicio está momentáneamente sobrecargado. Vuelve a intentarlo en unos "
          "instantes, o escribe a info@csasrl.it.",
}


def _upstream_error(language: str) -> str:
    return _UPSTREAM_ERROR_MESSAGE.get(language, _UPSTREAM_ERROR_MESSAGE["en"])


def _extract_product_images(
    sources: list[Source], message: str = "", history: list | None = None
) -> list[ProductImage]:
    """
    Images to show alongside the answer (max 2).

    Dimension drawings come first: when the answer was built from a datasheet's
    dimensions page, the quoted drawing is what makes its A/B/C letters mean
    something. The drawing must belong to the model the question names — the
    same registry lookup retrieval uses, on the same expanded search query, so
    a follow-up ("what do the letters mean?") keeps its conversation's model.
    Family stock photos fill any remaining slot.
    """
    named = tuple(find_model_sources(build_search_query(message, history))) if message else ()
    raw = get_dimension_drawings(sources, named)
    if len(raw) < 2:
        families = [s.product_family for s in sources if s.product_family]
        valve_models = [s.valve_model for s in sources if s.valve_model]
        seen = {item["url"] for item in raw}
        raw += [
            item for item in get_images_for_families(families, valve_models)
            if item["url"] not in seen
        ][: 2 - len(raw)]
    return [ProductImage(**item) for item in raw]


def _simli_headers() -> dict[str, str]:
    return {
        "x-simli-api-key": SIMLI_API_KEY,
        "content-type": "application/json",
    }


async def _fetch_simli_session_token(request: AvatarSessionRequest) -> str:
    if not SIMLI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="SIMLI_API_KEY is not configured on the server. Add it to .env or Render env vars.",
        )

    payload = {
        "faceId": request.face_id,
        "maxSessionLength": request.max_session_length,
        "maxIdleTime": request.max_idle_time,
        "handleSilence": request.handle_silence,
        "model": request.model,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(
                f"{SIMLI_API_BASE}/compose/token",
                headers=_simli_headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Simli token request failed: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Simli token error: {response.status_code} {response.text}",
        )

    data = response.json()
    session_token = data.get("session_token")
    if not session_token:
        raise HTTPException(status_code=502, detail="Simli token response did not include session_token.")
    return session_token


async def _fetch_simli_ice_servers() -> list[SimliIceServer]:
    if not SIMLI_API_KEY:
        return []

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(
                f"{SIMLI_API_BASE}/compose/ice",
                headers={"x-simli-api-key": SIMLI_API_KEY},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Simli ICE request failed: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Simli ICE error: {response.status_code} {response.text}",
        )

    return [SimliIceServer(**item) for item in response.json()]


async def _generate_chat_answer(request: ChatRequest, req: Request) -> tuple[str, str]:
    sources, detected_lang = await retrieve(
        query=request.message,
        language_hint=request.language,
        history=request.history,
    )

    await log_query(
        query=request.message,
        session_id=request.session_id,
        language=detected_lang,
        source_ip=req.client.host if req.client else None,
    )

    context_str = build_context_string(sources, detected_lang)
    system_prompt = build_system_prompt(context_str, detected_lang)

    completion = await async_oai.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        messages=_build_messages(system_prompt, request.history, request.message),
        temperature=0.2,
    )

    answer = sanitize_links(completion.choices[0].message.content or "", _allowed_links(sources))
    return answer, detected_lang


# ---------------------------------------------------------------------------
# Root — serve the chat widget UI
# ---------------------------------------------------------------------------
_WIDGET_PATH = pathlib.Path(__file__).parent.parent / "widget" / "chatbot.html"


@app.get("/")
async def root():
    # Il browser deve ricontrollare a ogni visita se la pagina e' cambiata.
    # Senza un Cache-Control il browser inventa una scadenza propria dalla data
    # del file: dopo un aggiornamento del widget i visitatori restavano con la
    # versione precedente per ore, e il pulsante ridisegnato non arrivava mai.
    # L'ETag rende la ricontrollata quasi gratuita (304, nessun corpo).
    return FileResponse(
        _WIDGET_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": CHAT_MODEL}


# ---------------------------------------------------------------------------
# POST /api/chat — synchronous JSON response
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    detected_lang = request.language or "en"
    try:
        # Retrieve relevant chunks (includes optional GPT reranking)
        sources, detected_lang = await retrieve(
            query=request.message,
            language_hint=request.language,
            history=request.history,
        )

        # Log the query (non-blocking)
        await log_query(
            query=request.message,
            session_id=request.session_id,
            language=detected_lang,
            source_ip=req.client.host if req.client else None,
        )

        # Build system prompt with context
        context_str = build_context_string(sources, detected_lang)
        system_prompt = build_system_prompt(context_str, detected_lang)

        # Call GPT-4o mini (with conversation history for follow-up context)
        completion = await async_oai.chat.completions.create(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            messages=_build_messages(system_prompt, request.history, request.message),
            temperature=0.2,
        )
    except Exception as exc:
        # A visitor gets a sentence they can act on rather than a 500 page.
        logger.exception("chat: upstream failure for session=%s", request.session_id)
        return ChatResponse(
            answer=_upstream_error(detected_lang),
            sources=[],
            detected_language=detected_lang,
            images=[],
        )

    answer = sanitize_links(completion.choices[0].message.content or "", _allowed_links(sources))
    images = _extract_product_images(sources, request.message, request.history)

    return ChatResponse(
        answer=answer,
        sources=sources,
        detected_language=detected_lang,
        images=images,
    )


@app.post("/api/avatar/session", response_model=AvatarSessionResponse)
async def avatar_session(request: AvatarSessionRequest):
    session_token = await _fetch_simli_session_token(request)
    ice_servers = await _fetch_simli_ice_servers()
    return AvatarSessionResponse(
        avatar_provider="simli",
        face_id=request.face_id,
        session_token=session_token,
        websocket_url=SIMLI_P2P_WEBSOCKET_URL,
        ice_servers=ice_servers,
        max_session_length=request.max_session_length,
        max_idle_time=request.max_idle_time,
    )


@app.post("/api/avatar/tts")
async def avatar_tts(request: AvatarTTSRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server. Add it to .env or Render env vars.",
        )

    instructions = request.instructions or (
        "Parla in italiano con voce femminile naturale, tono professionale ma caldo, "
        "ritmo medio e dizione chiara."
    )

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        rate_state = None
        async with async_oai.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=request.voice_id,
            input=request.text,
            instructions=instructions,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes(4096):
                if chunk:
                    converted, rate_state = audioop.ratecv(chunk, 2, 1, 24000, 16000, rate_state)
                    if converted:
                        yield converted

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Sample-Rate": "16000",
            "X-Audio-Channels": "1",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/chat/stream — SSE streaming response
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    detected_lang = request.language or "en"
    try:
        sources, detected_lang = await retrieve(
            query=request.message,
            language_hint=request.language,
            history=request.history,
        )

        # Log the query (non-blocking)
        await log_query(
            query=request.message,
            session_id=request.session_id,
            language=detected_lang,
            source_ip=req.client.host if req.client else None,
        )
    except Exception:
        # Retrieval failed before a single token was sent: reply with one
        # readable message over the same event stream the widget expects.
        logger.exception("chat_stream: retrieval failed for session=%s", request.session_id)

        async def failure_generator() -> AsyncGenerator[dict, None]:
            yield {"event": "metadata", "data": json.dumps(
                {"type": "metadata", "sources": [], "detected_language": detected_lang}
            )}
            yield {"event": "token", "data": json.dumps({"token": _upstream_error(detected_lang)})}
            yield {"event": "done", "data": json.dumps({"type": "done"})}

        return EventSourceResponse(failure_generator())

    context_str = build_context_string(sources, detected_lang)
    system_prompt = build_system_prompt(context_str, detected_lang)
    images = _extract_product_images(sources, request.message, request.history)

    async def event_generator() -> AsyncGenerator[dict, None]:
        # First event: metadata (sources, language)
        metadata_payload = {
            "type": "metadata",
            "sources": [s.model_dump() for s in sources],
            "detected_language": detected_lang,
        }
        yield {"event": "metadata", "data": json.dumps(metadata_payload)}

        # Links are checked as they stream: only link-sized fragments are held
        # back, so a URL is never shown before it has been verified against the
        # pages actually retrieved.
        sanitizer = StreamingLinkSanitizer(_allowed_links(sources))
        sent_any = False
        try:
            stream = await async_oai.chat.completions.create(
                model=CHAT_MODEL,
                max_tokens=MAX_TOKENS,
                messages=_build_messages(system_prompt, request.history, request.message),
                temperature=0.2,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    safe = sanitizer.feed(delta)
                    if safe:
                        sent_any = True
                        yield {"event": "token", "data": json.dumps({"token": safe})}

            tail = sanitizer.flush()
            if tail:
                yield {"event": "token", "data": json.dumps({"token": tail})}
        except Exception:
            # The stream can also die part-way through. Close it off with a
            # readable line instead of leaving the widget on a half sentence.
            logger.exception("chat_stream: generation failed for session=%s", request.session_id)
            note = _upstream_error(detected_lang)
            if sent_any:
                note = f"\n\n[{note}]"
            yield {"event": "token", "data": json.dumps({"token": note})}
            yield {"event": "done", "data": json.dumps({"type": "done"})}
            return

        # Send product images as a dedicated event before done
        if images:
            images_payload = {
                "type": "images",
                "images": [img.model_dump() for img in images],
            }
            yield {"event": "images", "data": json.dumps(images_payload)}

        yield {"event": "done", "data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_generator())
