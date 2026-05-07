"""Standalone upsert script — runs after url_map.json already exists."""
import os, json, time, sys
from pathlib import Path

# Load .env FIRST before any module-level os.environ access
from dotenv import load_dotenv
load_dotenv()

# Ensure OPENAI/PINECONE vars are set before importing ingest modules
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
assert os.environ.get("PINECONE_API_KEY"), "PINECONE_API_KEY not set"

from openai import OpenAI
from pinecone import Pinecone
from ingest.web_scraper import upsert_url_vectors, get_or_create_index, OUTPUT_PATH

oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = get_or_create_index(pc)

url_map = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(url_map)} URLs — starting upsert …")
n = upsert_url_vectors(url_map, oai, index)
print(f"Done. Upserted {n} URL vectors.")
