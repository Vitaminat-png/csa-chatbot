"""
api/links.py
------------
Guarantees every link in an answer is one that was actually retrieved.

The system prompt tells the model to copy URLs verbatim, and it mostly does —
but not reliably. Asked in Spanish about the Fortix strainer it returned
`/es/prodotto/filtro-csa-fortix-alta-efficiencia/`, "correcting" the Italian
word the real Spanish slug keeps (`efficienza`). The page it invented does not
exist. Across repeated runs the same question produced a good link and a broken
one, so instructions alone cannot close this.

Links are therefore checked against the URLs present in the retrieved context:
an unknown URL is either snapped to the one it was clearly meant to be, or the
link is unwrapped and only its text survives. A missing link is a small loss; a
404 sent to a customer is not.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional

# Markdown inline links: [label](url)
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")

# Below this similarity a wrong URL is not repaired, only unwrapped: a distant
# "closest match" would be a different product, which is worse than no link.
_SNAP_THRESHOLD = 0.90


def _normalize(url: str) -> str:
    return url.rstrip("/").lower()


def _closest(url: str, allowed: Iterable[str]) -> Optional[str]:
    """Return the allowed URL *url* was evidently meant to be, if any."""
    target = _normalize(url)
    best: Optional[str] = None
    best_ratio = 0.0
    for candidate in allowed:
        ratio = SequenceMatcher(None, target, _normalize(candidate)).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, candidate
    return best if best_ratio >= _SNAP_THRESHOLD else None


def sanitize_links(text: str, allowed: Iterable[str]) -> str:
    """
    Replace or unwrap every markdown link whose URL is not in *allowed*.

    Exact matches (ignoring case and a trailing slash) pass through unchanged.
    """
    allowed = [u for u in allowed if u]
    by_normal = {_normalize(u): u for u in allowed}

    def fix(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        exact = by_normal.get(_normalize(url))
        if exact:
            return f"[{label}]({exact})"
        snapped = _closest(url, allowed)
        if snapped:
            return f"[{label}]({snapped})"
        return label

    return _LINK_RE.sub(fix, text)


class StreamingLinkSanitizer:
    """
    Applies `sanitize_links` to a token stream without buffering the answer.

    Tokens pass straight through until a '[' appears; from there the text is
    held until the link closes, so a URL is never emitted before it has been
    checked. Only link-sized fragments are ever delayed.
    """

    def __init__(self, allowed: Iterable[str]) -> None:
        self._allowed = list(allowed)
        self._buffer = ""

    def feed(self, token: str) -> str:
        """Return the text safe to emit now, holding back any partial link."""
        self._buffer += token
        out = []

        while self._buffer:
            start = self._buffer.find("[")
            if start == -1:
                out.append(self._buffer)
                self._buffer = ""
                break

            out.append(self._buffer[:start])
            rest = self._buffer[start:]

            end = rest.find(")")
            if end == -1:
                # Link still open: keep waiting unless this cannot become one.
                if "]" in rest and "](" not in rest:
                    out.append(rest)
                    self._buffer = ""
                else:
                    self._buffer = rest
                break

            candidate = rest[: end + 1]
            self._buffer = rest[end + 1 :]
            out.append(sanitize_links(candidate, self._allowed))

        return "".join(out)

    def flush(self) -> str:
        """Emit whatever is still held once the stream ends."""
        remaining, self._buffer = self._buffer, ""
        return sanitize_links(remaining, self._allowed)
