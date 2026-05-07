"""
api/models.py
-------------
Pydantic schemas for the chat API.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for context")
    language: Optional[str] = Field(
        None,
        description="BCP-47 language hint (e.g. 'it', 'en', 'fr', 'es'). "
                    "Auto-detected from message if omitted.",
    )


class Source(BaseModel):
    source_file: str = Field(..., description="PDF filename or 'web_scraper'")
    page: Optional[int] = Field(None, description="Page number (PDFs only)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Cosine similarity score (0–1)")
    text_snippet: str = Field(..., description="First 200 chars of the chunk")
    url: Optional[str] = Field(None, description="Associated product URL if available")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer in the user's language")
    sources: list[Source] = Field(default_factory=list, description="Retrieved chunks used")
    detected_language: str = Field("en", description="Language detected/used for the answer")
