"""
ingest/pdf_extract.py
---------------------
Shared PDF page extraction for every ingest script.

pdfplumber's extract_text() is not enough for CSA's datasheets. Two failures
recur across the corpus and both destroy exactly the data the chatbot is asked
for:

* **Tables collapse into a single line.** A DN/Kv table renders as
  "DN (mm) 40 50 65 … Kv (m3/h) 40,6 40,6 68 …", which severs every value from
  its size, and on wide tables the last column is dropped outright — the XLC
  sizing page lost its DN 800 row entirely. Tables are therefore serialised one
  record per line, each line carrying its own labels.

* **Chart axes read as data.** The performance graphs leave runs of bare numbers
  ("25 20 15 10 5 0 1 2 3 4 …") in the text layer. They carry no attribute
  names, so they only dilute a chunk.

Measured over a sample of the 521 datasheet chunks indexed with plain
extract_text(), 11% were more than a fifth digits — flattened tables and axis
runs — including the DN/Kv table in XLC_ENG.pdf and the dimension table in
LYNX_SUB.pdf.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

import tiktoken

_AXIS_RUN = re.compile(r"(?:(?<=\s)|^)(?:[\d]+(?:[.,/]\d+)?\s+){5,}[\d]+(?:[.,/]\d+)?")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

# cl100k_base is the encoding used by text-embedding-3-small.
_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def _decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping token-based chunks."""
    tokens = _tokenize(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Table serialisation
# ---------------------------------------------------------------------------
# A "value group": an optional word prefix followed by a number, e.g. "80",
# "0,75", "CH 41", 'Threaded 1"'. Used to detect and undo merged rows.
#
# A range stays one group. Without that, the pressure ranges in VRCD_FF.pdf
# ("2-20" for 2 to 20 bar) were read as two values and the row was split into a
# nonsensical "2" row and a "20" row — turning a correct table into a wrong one.
#
# A trailing capital or asterisk belongs to its number: "Flanged 150R" is the
# reduced-flange size, distinct from "Flanged 150", and dropping the R while
# splitting handed the 150R its sibling's identity — exactly the ambiguity the
# split exists to remove. "1100*" keeps its footnote marker the same way.
_NUMBER = r"\d+(?:[.,]\d+)?"
_VALUE_GROUP = re.compile(
    rf"[A-Za-zÀ-ÿ]*\s*{_NUMBER}(?:\s*[-–÷/]\s*{_NUMBER})*[A-Z]?[\"'″]?\*?"
)


def _value_groups(cell: str) -> list[str]:
    """Split a cell into the value groups it contains."""
    return [m.group(0).strip() for m in _VALUE_GROUP.finditer(cell)]


def _split_merged_row(row: list[str]) -> Optional[list[list[str]]]:
    """
    Undo a row that holds several rows' worth of values, or return None.

    Where a datasheet's table has no ruling line between its data rows,
    pdfplumber returns them as one row whose every cell holds all the values:

        ['Threaded 1" Threaded 2"', '80 110', '167 226', 'CH 41 CH 65']

    which reads as a single size with two of every dimension. The split is only
    attempted when *every* filled cell yields the same number of value groups —
    a correctly extracted row has cells of differing shape ("Body" alongside
    "ductile cast iron GJS 450-10"), so it never satisfies this and is left
    untouched.
    """
    filled = [c for c in row if c]
    if len(filled) < 2:
        return None

    counts = {len(_value_groups(c)) for c in filled}
    if len(counts) != 1:
        return None

    n = counts.pop()
    if n < 2:
        return None

    return [
        [(_value_groups(cell)[i] if cell else "") for cell in row]
        for i in range(n)
    ]


def _split_merged_rows(grid: list[list[str]]) -> list[list[str]]:
    """
    Expand merged data rows back into one row per record.

    Only fully-filled rows are candidates. An earlier version gated on table
    size (<=3 rows) instead, on the theory that larger tables are ruled and
    extract correctly — but the CYCLOPS/GOLIA dimension tables are 10-row
    tables whose middle rows ARE fused ("Flanged 100 Flanged 150R" with
    "Weight Kg = 21,5 34"), and the gate skipped them: asked for the 150R's
    weight, the bot read 57 from another fused row. Full-filledness is the
    actual signature of a fused data row — the rows the size gate was
    protecting (APOLLO's continuation rows, diagram debris) are mostly-empty,
    so they fail this test and stay untouched, while a census of all 7637
    data rows in the corpus shows every fully-filled uniform-count row really
    is two records fused.
    """
    out: list[list[str]] = [grid[0]]
    for row in grid[1:]:
        split = _split_merged_row(row) if all(row) else None
        out.extend(split if split else [row])
    return out


def _clean_table(table: list[list]) -> list[list[str]]:
    """
    Normalise cells and drop rows/columns that are entirely empty.

    Cell text is flattened to a single line: header cells in the dimension
    tables wrap ("DN\\n(mm)", "Peso\\n(Kg)"), and an embedded newline splits the
    one-record-per-line output mid-label, producing garbled fragments like
    "(mm) = 310; D".
    """
    grid = [[re.sub(r"\s+", " ", (c or "")).strip() for c in row] for row in table]
    grid = [row for row in grid if any(row)]
    if not grid:
        return []

    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    keep = [i for i in range(width) if any(row[i] for row in grid)]
    grid = [[row[i] for i in keep] for row in grid]
    return _split_merged_rows(grid)


def _is_numeric_like(cell: str) -> bool:
    """True for size-style headers: '40', '40/50', '1,5'."""
    return bool(cell) and bool(re.fullmatch(r"[\d\s,./\-]+", cell))


def _header_is_numeric(header: list[str]) -> bool:
    """
    True when the header row holds sizes rather than attribute names.

    This picks the table's orientation: numeric headers mean sizes run across
    the columns and the table is transposed, textual headers ('Componente',
    'Materiale standard') mean each row is already one record.
    """
    filled = [h for h in header if h]
    if not filled:
        return False
    numeric = sum(1 for h in filled if _is_numeric_like(h))
    return numeric * 2 >= len(filled)


def _row_labels(grid: list[list[str]], n_label_cols: int) -> list[str]:
    """
    Build one composite label per row, forward-filling the label hierarchy.

    CSA tables nest headings across the leading columns:

        Portata (l/s) | Valori consigliati | Min. | 0.6 | 1,1 | ...
                      |                    | Max. | 13  | 23  | ...
                      | Sfioro pressione   | Max. | 20  | 34  | ...

    Reading only the first cell loses every row whose label is inherited, so a
    blank cell keeps the value carried down from the row above. Setting a cell
    clears the deeper levels, otherwise 'Sfioro pressione' would still read
    'Valori consigliati'.
    """
    carry = [""] * n_label_cols
    labels: list[str] = []
    for row in grid:
        for col in range(n_label_cols):
            if row[col]:
                carry[col] = row[col]
                for deeper in range(col + 1, n_label_cols):
                    carry[deeper] = ""
        labels.append(" ".join(part for part in carry if part))
    return labels


def serialize_table(table: list[list], page_num: int) -> str:
    """
    Render a table as text that keeps each value tied to its column.

    CSA dimension tables are attribute-per-row and size-per-column, so the
    table is transposed: one line per size, listing every attribute for it.
    Tables that are not that shape fall back to pipe-delimited rows.
    """
    grid = _clean_table(table)
    if len(grid) < 2 or len(grid[0]) < 2:
        return ""

    # Layout scaffolding around the technical drawings is extracted as tables
    # holding only callout letters. Without numbers there is nothing to answer
    # from, and the noise would displace real content in a chunk.
    if not any(any(ch.isdigit() for ch in cell) for row in grid for cell in row):
        return ""

    n_cols = len(grid[0])

    # The header row names the size axis in column 0, then leaves the remaining
    # label columns blank before the first data value.
    first_data_col = next((i for i in range(1, n_cols) if grid[0][i]), 1)
    labels = _row_labels(grid, first_data_col)
    header = [grid[0][c] for c in range(first_data_col, n_cols)]

    if not labels[0] or not header:
        return f"[Table p.{page_num}]\n" + "\n".join(" | ".join(row) for row in grid)

    lines: list[str] = []
    if _header_is_numeric(header):
        # Sizes run across the columns (DN 40, 50, 65 …) and attributes down the
        # rows, so one line per size collects every attribute for that size.
        for col in range(first_data_col, n_cols):
            parts = [
                f"{labels[row_idx]} = {row[col]}"
                for row_idx, row in enumerate(grid)
                if row[col] and labels[row_idx]
            ]
            if parts:
                lines.append("; ".join(parts))
    else:
        # The header names attributes (Componente, Materiale standard …), so
        # each row is one record and is emitted as such.
        for row in grid[1:]:
            parts = [f"{labels[0]} = {row[0]}"] if row[0] else []
            parts += [
                f"{header[i]} = {row[first_data_col + i]}"
                for i in range(len(header))
                if row[first_data_col + i] and header[i]
            ]
            if parts:
                lines.append("; ".join(parts))

    if lines:
        return f"[Table p.{page_num}]\n" + "\n".join(lines)
    return f"[Table p.{page_num}]\n" + "\n".join(" | ".join(row) for row in grid)


def split_table_block(block: str) -> list[str]:
    """
    Split one serialised table into chunk-sized pieces along row boundaries.

    Every serialised row repeats its own labels ("DN (mm) = 800; Kv (m3/h) =
    10479"), so a row is self-describing and cutting between rows loses nothing.
    Cutting *within* a row does lose data: chunking a page as one blob split the
    DN/Kv table mid-way and the largest size, DN 800, ended up in a different
    chunk from the rest of the table — so a question about available sizes got
    an answer that stopped at DN 600.
    """
    header, *rows = block.split("\n")
    header_cost = len(_enc.encode(header)) + 1
    budget = CHUNK_SIZE - header_cost

    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for row in rows:
        cost = len(_enc.encode(row)) + 1
        if current and used + cost > budget:
            pieces.append("\n".join([header, *current]))
            current, used = [], 0
        current.append(row)
        used += cost
    if current:
        pieces.append("\n".join([header, *current]))
    return pieces


# ---------------------------------------------------------------------------
# Page text
# ---------------------------------------------------------------------------
def strip_axis_runs(text: str) -> str:
    """
    Remove long runs of bare numbers left behind by chart axes.

    The performance pages carry cavitation and head-loss graphs whose tick
    labels extract as text ("25 20 15 10 5 0 1 2 3 4 …"). They read as data but
    carry no attribute names, so they only dilute a chunk — and the real figures
    are already captured from the tables.
    """
    return re.sub(r"\s{2,}", " ", _AXIS_RUN.sub(" ", text)).strip()


def _outside(bboxes: list[tuple]) -> Callable:
    """Predicate keeping only page objects that lie outside every bbox."""
    def keep(obj) -> bool:
        cx = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
        cy = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
        for x0, top, x1, bottom in bboxes:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False
        return True
    return keep


def page_units(
    page,
    page_num: int,
    heading_fn: Optional[Callable[[object], str]] = None,
) -> list[str]:
    """
    Return the page's indexable units: prose chunks plus whole tables.

    Prose and tables are chunked separately so a table is never broken by prose
    spilling over a token boundary. Regions covered by a table we serialised are
    then removed from the prose: extract_text() renders the same table flattened
    into one line, and its flattened form silently drops the final column, so
    keeping both would let the model answer from the truncated copy.

    *heading_fn* optionally returns a page heading to prefix onto every unit —
    needed when one document covers several product ranges and a bare table
    would not say which range it belongs to.
    """
    units: list[str] = []
    table_blocks: list[str] = []
    covered: list[tuple] = []

    for table in page.find_tables():
        try:
            block = serialize_table(table.extract(), page_num)
        except Exception:
            continue
        if block:
            table_blocks.append(block)
            covered.append(table.bbox)

    source = page.filter(_outside(covered)) if covered else page
    prose = strip_axis_runs(re.sub(r"\s+", " ", source.extract_text() or ""))
    if prose:
        units.extend(chunk_text(prose))

    for block in table_blocks:
        if len(_enc.encode(block)) <= CHUNK_SIZE:
            units.append(block)
        else:
            units.extend(split_table_block(block))

    if heading_fn is not None:
        heading = heading_fn(page)
        if heading:
            units = [f"{heading}\n{unit}" for unit in units]

    return units
