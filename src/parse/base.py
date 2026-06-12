"""Shared parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from src.models.enums import Sect, SourceCorpus
from src.parse.identity import ID_DELIMITER, SOURCE_CORPORA
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "write_parquet",
    "read_csv_robust",
    "generate_source_id",
    "safe_int",
    "safe_str",
    "validate_enum_fields",
    "provenance_violations",
]

# Provenance columns every staging schema is expected to populate, mapped to the
# closed value-domain each must stay within. ``sect`` / ``source_corpus`` are the
# enum-namespaced provenance on the hadith/collection/mention schemas; ``source``
# is the free-text provenance label on the narrator-bio / network-edge schemas
# (no enum domain, but it must still be non-empty). A table carrying none of
# these columns (a non-canonical intermediate) is simply not provenance-checked.
_PROVENANCE_DOMAINS: dict[str, frozenset[str] | None] = {
    "sect": frozenset(s.value for s in Sect),
    "source_corpus": frozenset(s.value for s in SourceCorpus),
    "source": None,
}


def provenance_violations(table: pa.Table) -> list[str]:
    """Return provenance-tagging violations for *table* (empty list == clean).

    The conformance gate for da#83: a non-nullable schema column already rejects
    a *null* ``sect``/``source_corpus`` at construction, but an **empty string**
    slips through a non-nullable column, and a **wrong-domain** value
    (``source_corpus="sunna"`` typo, a ``sect`` outside the enum) is never caught
    anywhere. This checks both, for whichever provenance columns the table
    carries, so every parser's output is tagged consistently and in-domain.
    """
    problems: list[str] = []
    columns = set(table.column_names)
    for column, domain in _PROVENANCE_DOMAINS.items():
        if column not in columns:
            continue
        values = table.column(column)
        if values.null_count:
            problems.append(f"{column}: {values.null_count} null value(s)")
        non_null = values.drop_null()
        if not len(non_null):
            continue
        empty = pc.sum(pc.equal(pc.utf8_trim_whitespace(non_null), "")).as_py() or 0
        if empty:
            problems.append(f"{column}: {empty} empty/whitespace value(s)")
        if domain is not None:
            out_of_domain = sorted(
                {
                    v
                    for v in pc.unique(non_null).to_pylist()
                    if v is not None and v.strip() != "" and v not in domain
                }
            )
            if out_of_domain:
                problems.append(f"{column}: value(s) outside {sorted(domain)}: {out_of_domain}")
    return problems


def write_parquet(table: pa.Table, path: Path, schema: pa.Schema | None = None) -> Path:
    """Write PyArrow Table to Parquet with Snappy compression.

    Validate against schema if provided, then enforce the provenance-tagging
    contract (:func:`provenance_violations`) as a fail-fast invariant so no
    parser can ship staging data with an empty or out-of-domain ``sect`` /
    ``source_corpus`` / ``source``. Create parent dirs. Log stats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if schema is not None:
        table = table.cast(schema)
    violations = provenance_violations(table)
    if violations:
        msg = f"provenance conformance failed for {path.name}: " + "; ".join(violations)
        raise ValueError(msg)
    pq.write_table(table, path, compression="snappy")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("wrote_parquet", path=str(path), rows=table.num_rows, size_mb=round(size_mb, 2))
    return path


def read_csv_robust(path: Path, encoding: str = "utf-8", **kwargs: Any) -> tuple[pa.Table, str]:
    """Read CSV with fallback encoding chain. Returns (table, encoding_used).

    Try: specified -> utf-8-sig -> latin-1.
    """
    encodings = [encoding]
    if encoding != "utf-8-sig":
        encodings.append("utf-8-sig")
    if "latin-1" not in encodings:
        encodings.append("latin-1")

    for enc in encodings:
        try:
            read_options = pcsv.ReadOptions(encoding=enc)
            table = pcsv.read_csv(str(path), read_options=read_options, **kwargs)
            logger.info("csv_read", path=str(path), encoding=enc, rows=table.num_rows)
            return table, enc
        except Exception:  # noqa: BLE001
            logger.warning("csv_encoding_failed", path=str(path), encoding=enc)
            continue

    msg = f"Failed to read CSV {path} with any encoding: {encodings}"
    raise ValueError(msg)


def generate_source_id(corpus: str, collection: str, *parts: int | str) -> str:
    """Generate a deterministic, corpus-namespaced ``source_id``.

    Example: ``generate_source_id("lk", "bukhari", 1, 1)`` -> ``"lk:bukhari:1:1"``

    *corpus* MUST be a known source corpus (:data:`identity.SOURCE_CORPORA`); an
    unknown corpus is a collision hazard (ids only stay unique across sources
    because the corpus namespaces them), so it fails fast. The canonical grammar
    and the graph node-id rules live in :mod:`src.parse.identity`.
    """
    if corpus not in SOURCE_CORPORA:
        msg = (
            f"unknown source corpus {corpus!r}; source_id must be namespaced by a "
            f"known corpus (one of {sorted(SOURCE_CORPORA)}) to stay collision-safe"
        )
        raise ValueError(msg)
    segments = [corpus, collection, *[str(p) for p in parts]]
    return ID_DELIMITER.join(segments)


def safe_int(value: Any) -> int | None:
    """Safely convert to int. Return None on failure."""
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):  # fmt: skip
        return None


def safe_str(value: Any) -> str | None:
    """Convert to stripped string. Return None if empty/NaN/None."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    return s


def validate_enum_fields(
    table: pa.Table,
    field_name: str,
    allowed_values: set[str],
) -> list[str]:
    """Validate string column values against allowed set.

    Returns list of invalid values found. Logs warnings.
    """
    column = table.column(field_name)
    unique_values: list[str | None] = column.unique().to_pylist()
    invalid = [v for v in unique_values if v is not None and v not in allowed_values]
    if invalid:
        logger.warning(
            "invalid_enum_values",
            field=field_name,
            invalid_values=invalid,
            allowed=sorted(allowed_values),
        )
    return invalid
