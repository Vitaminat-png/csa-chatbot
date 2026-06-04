"""
api/models.py
-------------
Pydantic schemas for the chat API.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for context")
    language: Optional[str] = Field(
        None,
        description="BCP-47 language hint (e.g. 'it', 'en', 'fr', 'es'). "
                    "Auto-detected from message if omitted.",
    )
    history: Optional[list[HistoryMessage]] = Field(
        default_factory=list,
        description="Conversation history (up to 10 messages) for follow-up context",
    )


class Source(BaseModel):
    source_file: str = Field(..., description="PDF filename or 'web_scraper'")
    page: Optional[int] = Field(None, description="Page number (PDFs only)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Cosine similarity score (0–1)")
    text_snippet: str = Field(..., description="First 200 chars of the chunk")
    url: Optional[str] = Field(None, description="Associated product URL if available")
    product_family: Optional[str] = Field(None, description="Product family from Pinecone metadata")
    valve_model: Optional[str] = Field(None, description="Valve model from Pinecone metadata")


class ProductImage(BaseModel):
    url: str = Field(..., description="Image src URL")
    alt: str = Field(..., description="Alt text for the image")
    product_name: str = Field(..., description="Human-readable product name")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer in the user's language")
    sources: list[Source] = Field(default_factory=list, description="Retrieved chunks used")
    detected_language: str = Field("en", description="Language detected/used for the answer")
    images: list[ProductImage] = Field(default_factory=list, description="Product images for mentioned families")


class AvatarRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for context")
    language: Optional[str] = Field(
        None,
        description="BCP-47 language hint (e.g. 'it', 'en'). Auto-detected from message if omitted.",
    )
    answer_text: Optional[str] = Field(
        None,
        description="Optional precomputed chatbot answer from the CSA backend",
    )
    detected_language: Optional[str] = Field(
        None,
        description="Optional language detected by the CSA backend",
    )
    history: Optional[list[HistoryMessage]] = Field(
        default_factory=list,
        description="Conversation history for chatbot context",
    )
    face_id: str = Field(..., min_length=1, description="Selected avatar face identifier")
    voice_id: str = Field(..., min_length=1, description="Selected avatar voice identifier")
    provider: str = Field("d-id", description="Avatar provider to use")


class AvatarVideoResponse(BaseModel):
    answer: str = Field(..., description="Chatbot answer used for avatar generation")
    detected_language: str = Field(..., description="Detected language of chatbot answer")
    avatar_provider: str = Field(..., description="Avatar provider used")
    face_id: str = Field(..., description="Avatar face identifier")
    voice_id: str = Field(..., description="Voice identifier")
    talk_id: str = Field(..., description="Provider talk/video generation identifier")
    status: str = Field(..., description="Current provider status")
    video_url: Optional[str] = Field(None, description="Generated avatar video URL when ready")
    estimated_latency_seconds: Optional[int] = Field(
        None,
        description="Estimated end-to-end latency for the current provider workflow",
    )
