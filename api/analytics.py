"""
api/analytics.py
----------------
Analytics tracking for chatbot queries.

Provides:
  log_query()                        — async, fire-and-forget logger (non-blocking)
  GET /api/analytics/top-queries     — top N most frequent queries
  GET /api/analytics/daily-stats     — query counts per day (last 30 days)

Storage: data/analytics_log.jsonl  (one JSON object per line, append-only)
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.admin_auth import require_admin_token

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
_LOG_FILE = _DATA_DIR / "analytics_log.jsonl"


def _ensure_log_file() -> None:
    """Create the data dir and log file if they don't exist yet."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _LOG_FILE.exists():
        _LOG_FILE.touch()


# ---------------------------------------------------------------------------
# Non-blocking logger
# ---------------------------------------------------------------------------
def _write_entry(entry: dict) -> None:
    """Synchronous write — runs in a thread pool so it doesn't block the event loop."""
    _ensure_log_file()
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def log_query(
    query: str,
    session_id: Optional[str],
    language: str,
    source_ip: Optional[str] = None,
) -> None:
    """
    Fire-and-forget: schedule the disk write in the default thread pool
    so neither /api/chat nor /api/chat/stream is blocked.
    """
    entry = {
        "query": query,
        "session_id": session_id,
        "language": language,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": source_ip,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _write_entry, entry)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _read_all_entries() -> list[dict]:
    """Read every valid JSON line from the log file."""
    if not _LOG_FILE.exists():
        return []
    entries: list[dict] = []
    with _LOG_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip malformed lines
    return entries


@router.get("/top-queries", dependencies=[Depends(require_admin_token)])
async def top_queries(limit: int = Query(default=20, ge=1, le=100)):
    """Return the top N most-frequent queries (case-insensitive, stripped)."""
    loop = asyncio.get_event_loop()
    entries = await loop.run_in_executor(None, _read_all_entries)

    counter: Counter = Counter()
    for e in entries:
        normalized = e.get("query", "").lower().strip()
        if normalized:
            counter[normalized] += 1

    top = [
        {"query": q, "count": c}
        for q, c in counter.most_common(limit)
    ]
    return {"total_queries": len(entries), "top_queries": top}


@router.get("/daily-stats", dependencies=[Depends(require_admin_token)])
async def daily_stats():
    """Return query counts grouped by day for the last 30 days."""
    loop = asyncio.get_event_loop()
    entries = await loop.run_in_executor(None, _read_all_entries)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    daily: defaultdict[str, int] = defaultdict(int)

    for e in entries:
        ts_raw = e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts >= cutoff:
                day = ts.date().isoformat()  # "YYYY-MM-DD"
                daily[day] += 1
        except (ValueError, TypeError):
            pass

    result = sorted(
        [{"date": d, "count": c} for d, c in daily.items()],
        key=lambda x: x["date"],
    )
    return {"period_days": 30, "daily_stats": result}
