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
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from api.models import ChatRequest, ChatResponse, Source
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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHAT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024

async_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)


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
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    # Retrieve relevant chunks
    sources, detected_lang = retrieve(
        query=request.message,
        language_hint=request.language,
    )

    # Build system prompt with context
    context_str = build_context_string(sources, detected_lang)
    system_prompt = build_system_prompt(context_str, detected_lang)

    # Call GPT-4o mini
    completion = await async_oai.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content or ""

    return ChatResponse(
        answer=answer,
        sources=sources,
        detected_language=detected_lang,
    )


# ---------------------------------------------------------------------------
# POST /api/chat/stream — SSE streaming response
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    sources, detected_lang = retrieve(
        query=request.message,
        language_hint=request.language,
    )
    context_str = build_context_string(sources, detected_lang)
    system_prompt = build_system_prompt(context_str, detected_lang)

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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ],
            temperature=0.2,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"event": "token", "data": json.dumps({"token": delta})}

        yield {"event": "done", "data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_generator())
