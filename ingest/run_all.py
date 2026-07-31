"""
ingest/run_all.py
-----------------
Orchestrator: rebuilds the whole Pinecone index from the local sources.

Steps
  1. Model registry  — maps valve model codes to datasheet filenames so exact
                       model lookups work (api/model_registry.json).
  2. PDF datasheets  — docs/*.pdf into the default namespace.
  3. XLC engineering — the current XLC 300/400 authority in it/en/fr/es, with
                       tables serialised so sizes survive.
  4. Site crawl      — content of every page reachable by navigating csasrl.it,
                       with per-language URLs.

Step 4 replaces the old sitemap-only web_scraper, which indexed URL slugs
without page content and included pages nothing links to. web_scraper.py is
kept for reference but is no longer part of this pipeline.

Usage:
    python -m ingest.run_all
    # or from repo root:
    python ingest/run_all.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.build_model_registry import main as registry_main
from ingest.pdf_ingest import main as pdf_main
from ingest.site_crawler import main as crawl_main
from ingest.xlc_ingest import main as xlc_main

STEPS = [
    ("Model registry (api/model_registry.json)", registry_main, False),
    ("PDF datasheets (docs/)", pdf_main, False),
    ("XLC engineering documents (xlc engeniering/)", xlc_main, False),
    ("Site crawl (csasrl.it reachable pages)", crawl_main, True),
]


def main() -> None:
    total = len(STEPS)
    for number, (label, step, is_async) in enumerate(STEPS, start=1):
        print("=" * 68)
        print(f"Step {number}/{total} — {label}")
        print("=" * 68)
        if is_async:
            asyncio.run(step())
        else:
            step()
        print()

    print("All ingest steps complete.")


if __name__ == "__main__":
    main()
