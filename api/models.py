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


class SimliIceServer(BaseModel):
    urls: str | list[str] = Field(..., description="STUN/TURN server URL(s)")
    username: Optional[str] = Field(None, description="ICE username if required")
    credential: Optional[str] = Field(None, description="ICE credential if required")


class AvatarSessionRequest(BaseModel):
    face_id: str = Field(..., min_length=1, description="Selected Simli face identifier")
    max_session_length: int = Field(
        900,
        ge=60,
        le=3600,
        description="Maximum Simli session length in seconds",
    )
    max_idle_time: int = Field(
        120,
        ge=30,
        le=900,
        description="Maximum idle time before Simli closes the session",
    )
    handle_silence: bool = Field(
        True,
        description="Whether Simli should keep the avatar responsive during silence",
    )
    model: Optional[str] = Field(
        "fasttalk",
        description="Optional Simli realtime model (e.g. 'fasttalk')",
    )


class AvatarSessionResponse(BaseModel):
    avatar_provider: str = Field(..., description="Avatar provider used")
    face_id: str = Field(..., description="Selected Simli face identifier")
    session_token: str = Field(..., description="Temporary Simli session token")
    websocket_url: str = Field(..., description="WebSocket URL for Simli WebRTC signaling")
    ice_servers: list[SimliIceServer] = Field(
        default_factory=list,
        description="Temporary ICE server credentials for the browser RTCPeerConnection",
    )
    max_session_length: int = Field(..., description="Session length requested")
    max_idle_time: int = Field(..., description="Idle timeout requested")


class AvatarTTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2500, description="Text chunk to synthesize")
    voice_id: str = Field(
        "coral",
        min_length=1,
        description="OpenAI TTS voice identifier",
    )
    language: Optional[str] = Field(
        "it",
        description="Target language hint for speaking style",
    )
    instructions: Optional[str] = Field(
        None,
        description="Optional speaking instructions for the TTS model",
    )
