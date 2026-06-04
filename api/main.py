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

import json
import os
import pathlib
from base64 import b64encode
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from api.analytics import log_query, router as analytics_router
from api.feedback import router as feedback_router
from api.models import AvatarRequest, AvatarVideoResponse, ChatRequest, ChatResponse, ProductImage, Source
from api.product_images import get_images_for_families
from api.prompt import build_system_prompt
from api.retrieval import build_context_string, detect_language, retrieve

load_dotenv()

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
    "http://127.0.0.1:5500",   # VS Code Live Server default
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
    app.mount("/static", StaticFiles(directory=str(_STATIC_PATH)), name="static")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
D_ID_API_KEY = os.environ.get("D_ID_API_KEY", "")
CHAT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
MAX_HISTORY = 10  # max conversation turns to include for context

async_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _build_messages(system_prompt: str, history, current_message: str) -> list[dict]:
    """Build the messages list for GPT, prepending conversation history."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for h in (history or [])[-MAX_HISTORY:]:
        role = h.role if h.role in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": h.content})
    msgs.append({"role": "user", "content": current_message})
    return msgs


def _extract_product_images(sources: list[Source]) -> list[ProductImage]:
    """
    Pull unique product_family and valve_model values from retrieved sources
    and return matching ProductImage objects (max 2).
    """
    families = [s.product_family for s in sources if s.product_family]
    valve_models = [s.valve_model for s in sources if s.valve_model]
    raw = get_images_for_families(families, valve_models)
    return [ProductImage(**item) for item in raw]


def _d_id_auth_header() -> str:
    encoded = b64encode(D_ID_API_KEY.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


async def _generate_chat_answer(request: ChatRequest, req: Request) -> tuple[str, str]:
    sources, detected_lang = await retrieve(
        query=request.message,
        language_hint=request.language,
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

    answer = completion.choices[0].message.content or ""
    return answer, detected_lang


# ---------------------------------------------------------------------------
# Root — serve the chat widget UI
# ---------------------------------------------------------------------------
_WIDGET_PATH = pathlib.Path(__file__).parent.parent / "widget" / "chatbot.html"


@app.get("/")
async def root():
    return FileResponse(_WIDGET_PATH, media_type="text/html")


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

    # Retrieve relevant chunks (includes optional GPT reranking)
    sources, detected_lang = await retrieve(
        query=request.message,
        language_hint=request.language,
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

    answer = completion.choices[0].message.content or ""
    images = _extract_product_images(sources)

    return ChatResponse(
        answer=answer,
        sources=sources,
        detected_language=detected_lang,
        images=images,
    )


@app.post("/api/avatar/respond", response_model=AvatarVideoResponse)
async def avatar_respond(request: AvatarRequest, req: Request):
    if request.provider != "d-id":
        raise HTTPException(status_code=400, detail="Only provider 'd-id' is supported in this POC.")
    if not D_ID_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="D_ID_API_KEY is not configured on the server. Add it to .env or Render env vars.",
        )

    if request.answer_text:
        answer = request.answer_text
        detected_lang = request.detected_language or request.language or "it"
    else:
        answer, detected_lang = await _generate_chat_answer(
            ChatRequest(
                message=request.message,
                session_id=request.session_id,
                language=request.language,
                history=request.history,
            ),
            req,
        )

    face_catalog = {
        "emma": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=1024&q=80",
        "sofia": "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?auto=format&fit=crop&w=1024&q=80",
        "giulia": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=1024&q=80",
    }
    source_url = face_catalog.get(request.face_id)
    if not source_url:
        raise HTTPException(status_code=400, detail="Unknown face_id.")

    payload = {
        "source_url": source_url,
        "script": {
            "type": "text",
            "input": answer,
            "provider": {
                "type": "microsoft",
                "voice_id": request.voice_id,
            },
        },
    }

    headers = {
        "Authorization": _d_id_auth_header(),
        "accept": "application/json",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            create_response = await client.post("https://api.d-id.com/talks", headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"D-ID request failed: {exc}") from exc

    if create_response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"D-ID error: {create_response.status_code} {create_response.text}",
        )

    data = create_response.json()

    return AvatarVideoResponse(
        answer=answer,
        detected_language=detected_lang,
        avatar_provider="d-id",
        face_id=request.face_id,
        voice_id=request.voice_id,
        talk_id=data.get("id", ""),
        status=data.get("status", "created"),
        video_url=data.get("result_url"),
        estimated_latency_seconds=12,
    )


@app.get("/api/avatar/status/{talk_id}", response_model=AvatarVideoResponse)
async def avatar_status(talk_id: str, face_id: str, voice_id: str, answer: str = "", detected_language: str = "it"):
    if not D_ID_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="D_ID_API_KEY is not configured on the server. Add it to .env or Render env vars.",
        )

    headers = {
        "Authorization": _d_id_auth_header(),
        "accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            status_response = await client.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"D-ID polling failed: {exc}") from exc

    if status_response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"D-ID polling error: {status_response.status_code} {status_response.text}",
        )

    data = status_response.json()

    return AvatarVideoResponse(
        answer=answer,
        detected_language=detected_language,
        avatar_provider="d-id",
        face_id=face_id,
        voice_id=voice_id,
        talk_id=talk_id,
        status=data.get("status", "processing"),
        video_url=data.get("result_url"),
        estimated_latency_seconds=12,
    )


# ---------------------------------------------------------------------------
# POST /api/chat/stream — SSE streaming response
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    sources, detected_lang = await retrieve(
        query=request.message,
        language_hint=request.language,
    )

    # Log the query (non-blocking)
    await log_query(
        query=request.message,
        session_id=request.session_id,
        language=detected_lang,
        source_ip=req.client.host if req.client else None,
    )

    context_str = build_context_string(sources, detected_lang)
    system_prompt = build_system_prompt(context_str, detected_lang)
    images = _extract_product_images(sources)

    async def event_generator() -> AsyncGenerator[dict, None]:
        # First event: metadata (sources, language)
        metadata_payload = {
            "type": "metadata",
            "sources": [s.model_dump() for s in sources],
            "detected_language": detected_lang,
        }
        yield {"event": "metadata", "data": json.dumps(metadata_payload)}

        # Stream tokens
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
                yield {"event": "token", "data": json.dumps({"token": delta})}

        # Send product images as a dedicated event before done
        if images:
            images_payload = {
                "type": "images",
                "images": [img.model_dump() for img in images],
            }
            yield {"event": "images", "data": json.dumps(images_payload)}

        yield {"event": "done", "data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_generator())
