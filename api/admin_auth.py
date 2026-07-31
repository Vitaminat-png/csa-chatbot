"""
api/admin_auth.py
-----------------
Shared guard for the read-only reporting endpoints.

The analytics and feedback-stats endpoints were reachable by anyone who knew the
path. On the live deployment that exposes what visitors ask CSA — the query log
is commercial intelligence, and `last_negatives` carries whole questions and
answer snippets, which is wherever a visitor happened to type their name, site
or phone number.

Set ANALYTICS_TOKEN in the environment to require it. When the variable is unset
the endpoints stay open, so an existing local setup keeps working, but the app
logs a warning at import so the gap is not silent in production.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

ANALYTICS_TOKEN = os.environ.get("ANALYTICS_TOKEN", "")

if not ANALYTICS_TOKEN:
    logger.warning(
        "ANALYTICS_TOKEN is not set — /api/analytics/* and /api/feedback/stats are "
        "readable by anyone. Set it in the environment to require a token."
    )


async def require_admin_token(x_analytics_token: str = Header(default="")) -> None:
    """
    Reject the request unless it carries the configured token.

    Compared with secrets.compare_digest so a wrong token cannot be recovered by
    timing the response.
    """
    if not ANALYTICS_TOKEN:
        return
    if not x_analytics_token or not secrets.compare_digest(
        x_analytics_token, ANALYTICS_TOKEN
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Analytics-Token header.")
