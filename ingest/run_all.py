"""
ingest/run_all.py
-----------------
Orchestrator: runs pdf_ingest then web_scraper in sequence.

Usage:
    python -m ingest.run_all
    # or from repo root:
    python ingest/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.pdf_ingest import main as pdf_main
from ingest.web_scraper import main as scraper_main


def main() -> None:
    print("=" * 60)
    print("Step 1/2 — PDF ingest")
    print("=" * 60)
    pdf_main()

    print()
    print("=" * 60)
    print("Step 2/2 — Web scraper (csasrl.it sitemaps)")
    print("=" * 60)
    scraper_main()

    print()
    print("All ingest steps complete.")


if __name__ == "__main__":
    main()
