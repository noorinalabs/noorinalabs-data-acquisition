"""Deterministic lexical PARALLEL_OF detection (da#100).

Detects parallel hadiths — **intra-sunni, intra-shia, and cross-sect** — by
normalized-token similarity over the matn, and materializes
``parallel_links.parquet`` (``PARALLEL_LINKS_SCHEMA``): the same artifact the
graph ``PARALLEL_OF`` loader (``src/graph/load_edges.py``) already consumes. This
directly feeds isnad-graph#964 (Browse Parallels, which must span all three
relationship kinds, not exclusively cross-sect).

Why a second detector alongside ``src/resolve/dedup.py``
--------------------------------------------------------
``dedup.py`` is the high-recall PRODUCTION path: multilingual
sentence-transformer embeddings + a FAISS index. It needs the model (a network
download) and non-trivial compute, and it *degrades to an empty output* when
those deps are unavailable — so it cannot deterministically prove cross-sect
detection on the loaded slice or in CI. This module is the **deterministic,
dependency-light** detector: pure-Python normalized-token similarity that runs
offline, in CI, and on the live slice, so the mechanism is provable now. Recall
grows as the embedding path and more corpora land; both detectors write the
identical ``PARALLEL_LINKS_SCHEMA``, so the downstream loader and the graph
contract are unchanged.

Matching signal
---------------
Token-set Jaccard over the normalized matn — Arabic via
:func:`src.utils.arabic.normalize_arabic` when an Arabic matn is present, else the
English matn lowercased. Scores are tiered into :class:`VariantType` by the same
thresholds ``dedup.py`` uses. ``cross_sect`` is read from the hadith ``sect``
column (authoritative), not inferred from the corpus name.

Scale note
----------
The current pairing is exhaustive O(n²), which is correct and ample for the
loaded slice. For full-corpus volume the same signal blocks on shared rare
tokens (an inverted index over significant tokens) before scoring — the
embedding/FAISS path in ``dedup.py`` is the production answer at that scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import VariantType
from src.resolve.schemas import PARALLEL_LINKS_SCHEMA
from src.utils.arabic import is_arabic, normalize_arabic
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

__all__ = ["detect_parallels", "run"]

# Tier thresholds — aligned with dedup.py ``_classify_pair`` so the two detectors
# label a pair the same way.
_VERBATIM_MIN = 0.90
_CLOSE_PARAPHRASE_MIN = 0.80
# Default minimum similarity to record a pair at all. Lexical Jaccard is stricter
# than cosine on embeddings, so the floor sits below the thematic tier ceiling;
# configurable per call.
_DEFAULT_THRESHOLD = 0.50


def _classify_pair(score: float) -> VariantType:
    """Classify a similarity score into a variant tier (matches dedup.py)."""
    if score >= _VERBATIM_MIN:
        return VariantType.VERBATIM
    if score >= _CLOSE_PARAPHRASE_MIN:
        return VariantType.CLOSE_PARAPHRASE
    return VariantType.THEMATIC


def _tokenize(matn_ar: str | None, matn_en: str | None) -> frozenset[str]:
    """Normalized token set for a hadith matn.

    Prefers the Arabic matn (normalized via ``normalize_arabic``); falls back to
    the English matn lowercased. Returns an empty set when neither is usable, in
    which case the hadith is excluded from pairing.
    """
    if matn_ar and is_arabic(matn_ar):
        normalized = normalize_arabic(matn_ar)
    elif matn_en and matn_en.strip():
        normalized = matn_en.lower()
    else:
        return frozenset()
    return frozenset(tok for tok in normalized.split() if tok)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


def _load_hadith_rows(staging_dir: Path) -> list[dict[str, object]]:
    """Load (source_id, matn_ar, matn_en, sect) for every staged hadith."""
    hadith_files = sorted(staging_dir.glob("**/hadiths_*.parquet"))
    rows: list[dict[str, object]] = []
    for fpath in hadith_files:
        table = pq.read_table(fpath, columns=["source_id", "matn_ar", "matn_en", "sect"])
        rows.extend(table.to_pylist())
    logger.info("parallels_loaded_hadiths", rows=len(rows), files=len(hadith_files))
    return rows


def _write_links(staging_dir: Path, records: Iterable[dict[str, object]]) -> Path:
    """Write the parallel_links.parquet table (empty table when no records)."""
    records = list(records)
    table = pa.table(
        {
            "hadith_id_a": pa.array([r["hadith_id_a"] for r in records], type=pa.string()),
            "hadith_id_b": pa.array([r["hadith_id_b"] for r in records], type=pa.string()),
            "similarity_score": pa.array(
                [r["similarity_score"] for r in records], type=pa.float32()
            ),
            "variant_type": pa.array([r["variant_type"] for r in records], type=pa.string()),
            "cross_sect": pa.array([r["cross_sect"] for r in records], type=pa.bool_()),
        },
        schema=PARALLEL_LINKS_SCHEMA,
    )
    output_path = staging_dir / "parallel_links.parquet"
    pq.write_table(table, output_path)
    return output_path


def detect_parallels(staging_dir: Path, *, threshold: float = _DEFAULT_THRESHOLD) -> Path:
    """Detect parallel hadiths by lexical matn similarity.

    Writes ``parallel_links.parquet`` (always — empty table when no pairs clear
    *threshold*) and returns its path. Pairs are canonically ordered
    (``hadith_id_a`` < ``hadith_id_b``) and symmetric duplicates collapsed;
    ``cross_sect`` is taken from the two hadiths' ``sect`` values.
    """
    rows = _load_hadith_rows(staging_dir)

    # Pre-tokenize once; drop hadiths with no usable matn.
    indexed: list[tuple[str, str | None, frozenset[str]]] = []
    for row in rows:
        sid = row.get("source_id")
        if not isinstance(sid, str) or not sid:
            continue
        matn_ar = row.get("matn_ar")
        matn_en = row.get("matn_en")
        sect = row.get("sect")
        tokens = _tokenize(
            matn_ar if isinstance(matn_ar, str) else None,
            matn_en if isinstance(matn_en, str) else None,
        )
        if not tokens:
            continue
        indexed.append((sid, sect if isinstance(sect, str) else None, tokens))

    records: list[dict[str, object]] = []
    cross_sect_count = 0
    for i in range(len(indexed)):
        id_i, sect_i, tok_i = indexed[i]
        for j in range(i + 1, len(indexed)):
            id_j, sect_j, tok_j = indexed[j]
            score = _jaccard(tok_i, tok_j)
            if score < threshold:
                continue
            # Canonical ordering for a stable, symmetric-dedup'd edge identity.
            id_a, id_b = (id_i, id_j) if id_i < id_j else (id_j, id_i)
            cross_sect = bool(sect_i and sect_j and sect_i != sect_j)
            if cross_sect:
                cross_sect_count += 1
            records.append(
                {
                    "hadith_id_a": id_a,
                    "hadith_id_b": id_b,
                    "similarity_score": float(score),
                    "variant_type": str(_classify_pair(score)),
                    "cross_sect": cross_sect,
                }
            )

    output_path = _write_links(staging_dir, records)
    logger.info(
        "parallels_detected",
        candidates=len(indexed),
        pairs=len(records),
        cross_sect=cross_sect_count,
        threshold=threshold,
        path=str(output_path),
    )
    return output_path


def run(staging_dir: Path, output_dir: Path) -> list[Path]:  # noqa: ARG001
    """Resolve-pipeline entry point. ``output_dir`` is accepted for interface
    symmetry with the other resolve stages; links are written into *staging_dir*
    next to the hadith parquet the graph loader reads."""
    return [detect_parallels(staging_dir)]
