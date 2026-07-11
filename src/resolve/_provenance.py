"""Detector provenance for ``parallel_links.parquet`` (da#378).

The composed ``parallel_links.parquet`` is written by two detectors — the
semantic ``dedup`` stage (embeddings + FAISS) and the deterministic ``parallels``
stage (lexical). Both emit the identical ``PARALLEL_LINKS_SCHEMA``, and a
zero-row table from either is **byte-identical** to a zero-row table from the
other. In particular a ``DEDUP_REQUIRE_ML=false`` degraded run — where ``dedup``
wrote an empty table ON PURPOSE because its ML deps were absent, while the
deterministic detector still ran — is byte-identical to a *true negative* where
``dedup`` ran on real embeddings and found no pairs. After a 7.5-hour run an
operator could not tell whether ``dedup`` ran at all (da#378).

This module makes the two states occupy **different bytes**: every write of
``parallel_links.parquet`` carries detector provenance in the parquet file's
key-value metadata, and the write helper :func:`write_parallel_links` *requires*
it. A provenance-less artifact is therefore unrepresentable through the sanctioned
writer — the conflation is closed at write time, not merely detected afterwards
(#928). :func:`read_provenance` is the discriminator: it returns
:attr:`DetectorStatus.RAN` for a true negative and
:attr:`DetectorStatus.DEGRADED_NO_ML` for a degraded run, so the two separate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

__all__ = [
    "PROVENANCE_METADATA_KEY",
    "DetectorProvenance",
    "DetectorStatus",
    "read_provenance",
    "write_parallel_links",
]

# The single parquet key-value-metadata key under which the provenance JSON is
# stored. Bytes, because parquet metadata is a ``dict[bytes, bytes]``.
PROVENANCE_METADATA_KEY = b"resolve.detector_provenance"


class DetectorStatus(StrEnum):
    """Why a detector's rows are what they are — the discriminator's alphabet.

    ``StrEnum`` per this repo's enum convention (``src/models/enums.py``) so it
    serialises to a plain string in the provenance JSON. The load-bearing
    distinction for da#378 is
    :attr:`RAN` (the detector executed its real algorithm; zero rows is a genuine
    negative) versus :attr:`DEGRADED_NO_ML` (``dedup`` skipped its algorithm and
    wrote an empty table on purpose) — the two states that were byte-identical.
    """

    RAN = "ran"
    """The detector executed its real algorithm. Zero rows is a *true negative*."""

    DEGRADED_NO_ML = "degraded_no_ml"
    """``dedup`` only: ML deps absent and ``DEDUP_REQUIRE_ML=false`` — empty on purpose."""

    NO_INPUT = "no_input"
    """``dedup`` only: no hadith files in staging (an upstream defect, not a negative)."""

    NO_TEXTS = "no_texts"
    """``dedup`` only: files present, but no row carried a non-empty English matn."""

    ERRORED = "errored"
    """The detector raised; ``run_all`` will exit non-zero (da#360). Rows are absent/partial."""

    NOT_RUN = "not_run"
    """The detector was not invoked this run (skipped via ``--from-step`` past it)."""


@dataclass(frozen=True)
class DetectorProvenance:
    """Which detectors produced ``parallel_links.parquet`` and in what state.

    Stamped into the parquet's key-value metadata so a zero-row artifact records
    *why* it is zero-row. ``semantic`` describes the ``dedup`` stage; the
    ``DEGRADED_NO_ML``/``NO_INPUT``/``NO_TEXTS`` statuses only ever apply there.
    ``deterministic`` describes the ``parallels`` stage, which is only ever
    :attr:`~DetectorStatus.RAN`, :attr:`~DetectorStatus.ERRORED` or
    :attr:`~DetectorStatus.NOT_RUN`.
    """

    semantic: DetectorStatus
    semantic_rows: int
    deterministic: DetectorStatus
    deterministic_rows: int

    def to_json_bytes(self) -> bytes:
        """Serialise to the UTF-8 JSON bytes stored in parquet metadata."""
        payload = asdict(self)
        payload["semantic"] = self.semantic.value
        payload["deterministic"] = self.deterministic.value
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> DetectorProvenance:
        """Inverse of :meth:`to_json_bytes`."""
        payload = json.loads(raw.decode("utf-8"))
        return cls(
            semantic=DetectorStatus(payload["semantic"]),
            semantic_rows=int(payload["semantic_rows"]),
            deterministic=DetectorStatus(payload["deterministic"]),
            deterministic_rows=int(payload["deterministic_rows"]),
        )


def write_parallel_links(
    table: pa.Table, output_path: Path, provenance: DetectorProvenance
) -> Path:
    """Write ``parallel_links.parquet`` with ``provenance`` stamped into its metadata.

    The single sanctioned writer of the artifact. ``provenance`` is a **required**
    argument: there is no code path through this helper that writes the table
    without declaring which detectors produced it, which is what makes the
    empty/true-negative conflation unrepresentable rather than after-the-fact
    detectable (da#378, #928).
    """
    existing = dict(table.schema.metadata or {})
    existing[PROVENANCE_METADATA_KEY] = provenance.to_json_bytes()
    stamped = table.replace_schema_metadata(existing)
    pq.write_table(stamped, output_path)
    return output_path


def read_provenance(path: Path) -> DetectorProvenance | None:
    """Read the detector provenance stamped in ``path``'s metadata, or ``None``.

    ``None`` means the artifact carries no provenance to read — the file is absent,
    or it predates da#378, or it was written by something other than
    :func:`write_parallel_links`. The absence guard mirrors :func:`_read_parallel_links`
    so a stubbed/skipped detector that wrote no file does not raise here. The
    discriminator for da#378: a true negative reads back
    :attr:`DetectorStatus.RAN`, a degraded run :attr:`DetectorStatus.DEGRADED_NO_ML`.
    """
    if not path.exists():
        return None
    schema = pq.read_schema(path)
    raw = (schema.metadata or {}).get(PROVENANCE_METADATA_KEY)
    if raw is None:
        return None
    return DetectorProvenance.from_json_bytes(raw)
