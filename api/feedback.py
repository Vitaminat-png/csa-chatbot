"""
api/feedback.py
---------------
Feedback router for CSA Chatbot.

Endpoints:
  POST /api/feedback        — save user feedback (thumbs up/down)
  GET  /api/feedback/stats  — aggregate stats + last 10 negative responses
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.admin_auth import require_admin_token

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
FEEDBACK_FILE = DATA_DIR / "feedback_log.jsonl"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: str          # "positive" | "negative"
    session_id: str = ""
    timestamp: str = ""


@router.post("")
async def post_feedback(req: FeedbackRequest):
    if req.rating not in ("positive", "negative"):
        raise HTTPException(
            status_code=400,
            detail="rating must be 'positive' or 'negative'",
        )

    _ensure_data_dir()

    record = {
        "query": req.query,
        "response_snippet": req.response[:200],
        "rating": req.rating,
        "session_id": req.session_id,
        "timestamp": req.timestamp or datetime.now(timezone.utc).isoformat(),
    }

    with FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "ok"}


# Posting feedback stays open — it is how the widget's thumbs work — but reading
# the aggregate back exposes visitors' questions and is guarded.
@router.get("/stats", dependencies=[Depends(require_admin_token)])
async def get_stats():
    if not FEEDBACK_FILE.exists():
        return {"positive": 0, "negative": 0, "total": 0, "last_negatives": []}

    records: list[dict] = []
    with FEEDBACK_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    positive = sum(1 for r in records if r.get("rating") == "positive")
    negative = sum(1 for r in records if r.get("rating") == "negative")
    last_negatives = [r for r in records if r.get("rating") == "negative"][-10:]

    return {
        "positive": positive,
        "negative": negative,
        "total": len(records),
        "last_negatives": last_negatives,
    }
