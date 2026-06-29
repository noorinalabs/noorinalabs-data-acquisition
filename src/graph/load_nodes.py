"""Neo4j node loading for all graph node types.

Batch UNWIND+MERGE loaders for Narrator, Hadith, Collection, Chain,
Grading, HistoricalEvent, and Location nodes.  Each loader reads from
staging/curated Parquet or YAML, validates rows, and merges into Neo4j
with explicit property SET (no ``SET n += row``) for Phase 4 safety.

Artifact-location contract (resolve -> load)
--------------------------------------------
Inputs split by *kind*, not by which loader consumes them:

* **staging dir** — raw, per-source parse outputs and resolve intermediates:
  ``hadiths_*``, ``collections_*``, ``narrator_mentions_*`` (Chain source),
  ``parallel_links``. These are the parse/resolve *working* artifacts.
* **curated dir** — the curated, resolved master tables and hand-maintained
  reference data: ``narrators_canonical.parquet`` (the canonical narrator
  master produced by the resolve stage), plus ``historical_events.yaml`` and
  ``locations.yaml``.

``narrators_canonical.parquet`` is the canonical narrator master *written by
the resolve stage* — ``disambiguate.run`` and ``bio_promote`` both emit it into
the resolve ``output_dir``, which ``src/cli.py`` maps to ``DATA_CURATED_DIR``.
The loader therefore reads it from ``curated_dir`` so writer and reader agree on
one location (da#112). It is NOT a raw staging intermediate like
``narrator_mentions_*`` (those stay in staging, read by the Chain loader).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from src.parse.composition import (
    canonical_matn_identity,
    is_canonical_hadith,
    is_cross_edition_dedup_source,
)
from src.parse.identity import (
    chain_node_id,
    collection_node_id,
    grading_node_id,
    hadith_node_id,
)
from src.utils.arabic import transliterate
from src.utils.grade import normalize_grade
from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = ["load_all_nodes", "LoadResult"]


@dataclass(frozen=True)
class LoadResult:
    """Outcome of loading a single node type."""

    node_type: str
    created: int
    merged: int
    skipped: int
    validation_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parquet_files(directory: Path, prefix: str) -> list[Path]:
    """Return sorted parquet files matching *prefix* in *directory*."""
    return sorted(directory.glob(f"{prefix}*.parquet"))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file and return row dicts."""
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _read_parquet_columns(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    """Read only *columns* from a Parquet file and return row dicts.

    Used by the cross-edition identity pre-pass so it can scan every hadith file
    for its matn/composition keys without materializing the full (wide) rows of
    the ~650k-row corpus.
    """
    table = pq.read_table(path, columns=columns)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _val(row: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from *row*, returning *default* for ``None``."""
    v = row.get(key)
    return default if v is None else v


def _effective_matn_ar(row: dict[str, Any]) -> str:
    """Arabic matn used for display *and* cross-edition identity.

    Sources that split isnad from matn populate ``matn_ar`` directly; those that
    only carry a combined body (halimbahae / open_hadith / bihar) populate
    ``full_text_ar``. Falling back to ``full_text_ar`` here keeps one rule for
    both the persisted node text (da#190) and the dedup identity, so a curated
    hadith and a sanadset duplicate are compared on the same basis.
    """
    matn_ar = _val(row, "matn_ar", "")
    if matn_ar and str(matn_ar).strip():
        return str(matn_ar)
    return str(_val(row, "full_text_ar", ""))


# Columns the cross-edition identity pre-pass needs from each hadith file.
_IDENTITY_COLUMNS = ["source_corpus", "collection_name", "matn_ar", "matn_en", "full_text_ar"]


def _build_curated_identity_index(files: list[Path]) -> set[str]:
    """Set of cross-edition identities occupied by the *curated* sources (da#220).

    Scans every hadith file and collects :func:`canonical_matn_identity` for each
    row that (a) is NOT from a cross-edition dedup source and (b) passes the
    per-source :func:`is_canonical_hadith` gate — i.e. exactly the curated
    traditions that will actually load. A dedup-source hadith whose identity is
    in this set duplicates an already-richer curated edition and is dropped
    ("curated wins"). Built once, up front, because a dedup-source file may sort
    before or after the curated files it must be checked against.
    """
    index: set[str] = set()
    for fp in files:
        for row in _read_parquet_columns(fp, _IDENTITY_COLUMNS):
            corpus = _val(row, "source_corpus", "")
            if is_cross_edition_dedup_source(corpus):
                continue
            if not is_canonical_hadith(corpus, _val(row, "collection_name", "")):
                continue
            identity = canonical_matn_identity(_effective_matn_ar(row), _val(row, "matn_en"))
            if identity is not None:
                index.add(identity)
    return index


def _narrator_name_en(row: dict[str, Any]) -> str:
    """Resolve a narrator's English display name, with a transliteration fallback.

    Almost no narrator records carry a sourced ``name_en`` (113 / 47,199 at
    P5W3 — da#159), leaving English search and display hollow. When a sourced
    English name is present we use it verbatim; otherwise we synthesize a
    deterministic Latin transliteration of ``name_ar`` (falling back to
    ``name_ar_normalized``) so every Narrator node has a non-empty, searchable
    English form.

    The fallback is applied at *load* time rather than persisted into
    ``narrators_canonical.parquet`` on purpose: the parquet ``name_en`` stays a
    pure provenance field (sourced-or-empty), which keeps ``bio_promote``'s
    only-when-missing back-fill idempotent and clobber-free, while the Neo4j
    node — what ``/graph``, search and narrator pages actually read — always has
    a display name. Legacy canonical files written before any English-name work
    are covered too, with no resolve re-run required.
    """
    sourced = _val(row, "name_en", "")
    if isinstance(sourced, str) and sourced.strip():
        return sourced
    name_ar = _val(row, "name_ar", "") or _val(row, "name_ar_normalized", "")
    return transliterate(name_ar) if name_ar else ""


# ---------------------------------------------------------------------------
# Narrator loader
# ---------------------------------------------------------------------------

_NARRATOR_MERGE = """\
UNWIND $batch AS row
MERGE (n:Narrator {id: row.id})
SET n.name_ar           = row.name_ar,
    n.name_en           = row.name_en,
    n.name_ar_normalized = row.name_ar_normalized,
    n.birth_year_ah     = row.birth_year_ah,
    n.death_year_ah     = row.death_year_ah,
    n.generation        = row.generation,
    n.gender            = row.gender,
    n.trustworthiness   = row.trustworthiness,
    n.aliases           = row.aliases,
    n.external_id       = row.external_id,
    n.source_ids        = row.source_ids,
    n.mention_count     = row.mention_count,
    n.source_corpus     = row.source_corpus,
    n.source_corpora    = row.source_corpora,
    n.sect_affiliation  = row.sect_affiliation
"""


def _load_narrators(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> LoadResult:
    """Load Narrator nodes from narrators_canonical.parquet.

    Reads from ``curated_dir`` — the canonical narrator master is a resolve-stage
    *output* (written by ``disambiguate.run`` / ``bio_promote`` into the resolve
    ``output_dir``, which the CLI maps to ``DATA_CURATED_DIR``), not a raw
    staging intermediate. See the module-level artifact-location contract (da#112).
    """
    path = curated_dir / "narrators_canonical.parquet"
    if _should_skip_file(path, staging_dir, skip_files):
        logger.info("narrators_skipped_incremental")
        return LoadResult("Narrator", 0, 0, 0)
    if not path.exists():
        if strict:
            msg = f"Missing required file: {path}"
            raise FileNotFoundError(msg)
        logger.warning("narrator_file_missing", path=str(path))
        return LoadResult("Narrator", 0, 0, 0)

    rows = _read_parquet_rows(path)
    batch: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0

    for i, row in enumerate(rows):
        cid = row.get("canonical_id")
        if not cid or not isinstance(cid, str) or not cid.startswith("nar:"):
            errors.append(f"row {i}: invalid canonical_id={cid!r}")
            skipped += 1
            continue
        batch.append(
            {
                "id": cid,
                "name_ar": _val(row, "name_ar", ""),
                "name_en": _narrator_name_en(row),
                "name_ar_normalized": _val(row, "name_ar_normalized"),
                "birth_year_ah": _val(row, "birth_year_ah"),
                "death_year_ah": _val(row, "death_year_ah"),
                "generation": _val(row, "generation"),
                "gender": _val(row, "gender"),
                "trustworthiness": _val(row, "trustworthiness"),
                "aliases": _val(row, "aliases", []),
                "external_id": _val(row, "external_id"),
                # Per-source provenance (``<corpus>:<bare-id>`` list) so a corpus is
                # auditable/removable on the graph itself, e.g.
                # ``MATCH (n:Narrator) WHERE any(s IN n.source_ids WHERE s STARTS WITH 'itqan:')``.
                "source_ids": _val(row, "source_ids", []),
                "mention_count": _val(row, "mention_count"),
                # Sect/corpus provenance (da#103). ``source_corpus`` defaults to ""
                # so the property is always present even for legacy canonical files
                # written before these columns existed; ``sect_affiliation`` defaults
                # to ``unknown`` for the same reason.
                "source_corpus": _val(row, "source_corpus", ""),
                "source_corpora": _val(row, "source_corpora", []),
                "sect_affiliation": _val(row, "sect_affiliation", "unknown"),
            }
        )

    created = client.execute_write_batch(_NARRATOR_MERGE, batch) if batch else 0
    merged = len(batch) - created
    logger.info(
        "narrators_loaded",
        created=created,
        merged=merged,
        skipped=skipped,
        errors=len(errors),
    )
    return LoadResult("Narrator", created, merged, skipped, errors)


# ---------------------------------------------------------------------------
# Hadith loader
# ---------------------------------------------------------------------------

# NOTE (#35, ratified P3W13): book_number / chapter_number / hadith_number are
# positional facts of a hadith's place *within a collection*, so they live on the
# APPEARS_IN edge (see load_edges.py ``_APPEARS_IN_QUERY``), NOT on the Hadith node.
# The Kafka ingest-platform normalize path already complies; this legacy loader is
# reconciled here. The staging schema still carries these columns — they feed the
# edge loader — they are simply no longer copied onto the node.
_HADITH_MERGE = """\
UNWIND $batch AS row
MERGE (n:Hadith {id: row.id})
SET n.matn_ar      = row.matn_ar,
    n.matn_en      = row.matn_en,
    n.isnad_raw_ar = row.isnad_raw_ar,
    n.isnad_raw_en = row.isnad_raw_en,
    n.grade        = row.grade,
    n.grade_normalized = row.grade_normalized,
    n.source_corpus = row.source_corpus,
    n.sect         = row.sect,
    n.collection_name = row.collection_name
"""


def _load_hadiths(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> LoadResult:
    """Load Hadith nodes from hadiths_*.parquet files."""
    files = _filter_parquet_files(_parquet_files(staging_dir, "hadiths_"), staging_dir, skip_files)
    if not files:
        if strict:
            msg = f"No hadiths_*.parquet files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("hadith_files_missing", dir=str(staging_dir))
        return LoadResult("Hadith", 0, 0, 0)

    total_created = 0
    total_skipped = 0
    total_deduped = 0
    all_errors: list[str] = []
    total_batch = 0

    # Cross-edition canonical-identity dedup (da#220 / Path B-B2): identities
    # occupied by the curated sources, against which dedup-source hadiths
    # (e.g. sanadset) are checked so the same tradition is not double-counted.
    # Built once up front because a dedup-source file may sort before the curated
    # files it must be checked against.
    curated_identities = _build_curated_identity_index(files)

    for fp in files:
        rows = _read_parquet_rows(fp)
        batch: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            sid = row.get("source_id")
            if not sid or not isinstance(sid, str):
                all_errors.append(f"{fp.name} row {i}: invalid source_id={sid!r}")
                total_skipped += 1
                continue
            source_corpus = _val(row, "source_corpus", "")
            # Canonical corpus composition (da#191): skip Hadith whose
            # (source_corpus, collection) duplicates the chosen canonical edition
            # — halimbahae/fawaz non-unique books, the mis Sahih Muslim matn copy.
            # One enforcement point so a fresh run_all (the production path) yields
            # the deduped graph without manual surgery.
            if not is_canonical_hadith(source_corpus, _val(row, "collection_name", "")):
                total_skipped += 1
                continue
            # Fall back to the raw full text when a source supplies no separated
            # matn: halimbahae / open_hadith / bihar populate ``full_text_ar``
            # only (they do not split isnad from matn). The loader persists
            # matn_ar — not full_text_ar — so without this the Hadith node lands
            # textless on the graph (da#190).
            matn_ar = _effective_matn_ar(row)
            # Cross-edition canonical-identity dedup (da#220): drop a dedup-source
            # hadith whose normalized matn already exists from a curated edition —
            # curated wins (richer metadata). Curated rows and dedup-source rows
            # with no curated twin fall straight through.
            if is_cross_edition_dedup_source(source_corpus):
                identity = canonical_matn_identity(matn_ar, _val(row, "matn_en"))
                if identity is not None and identity in curated_identities:
                    total_skipped += 1
                    total_deduped += 1
                    continue
            hid = hadith_node_id(sid)
            batch.append(
                {
                    "id": hid,
                    "matn_ar": matn_ar,
                    "matn_en": _val(row, "matn_en"),
                    "isnad_raw_ar": _val(row, "isnad_raw_ar"),
                    "isnad_raw_en": _val(row, "isnad_raw_en"),
                    "grade": _val(row, "grade"),
                    # Normalized display grade alongside the verbatim raw value: a
                    # corpus may store the grade as raw Arabic / mixed script,
                    # which is not a stable key to filter or display on (da#148).
                    "grade_normalized": normalize_grade(_val(row, "grade")),
                    "source_corpus": _val(row, "source_corpus", ""),
                    "sect": _val(row, "sect", ""),
                    "collection_name": _val(row, "collection_name", ""),
                    # book/chapter/hadith_number intentionally omitted — they belong
                    # on the APPEARS_IN edge, not the Hadith node (#35). See _HADITH_MERGE.
                }
            )
        if batch:
            total_created += client.execute_write_batch(_HADITH_MERGE, batch)
            total_batch += len(batch)

    merged = total_batch - total_created
    logger.info(
        "hadiths_loaded",
        files=len(files),
        created=total_created,
        merged=merged,
        skipped=total_skipped,
        cross_edition_deduped=total_deduped,
        curated_identities=len(curated_identities),
    )
    return LoadResult("Hadith", total_created, merged, total_skipped, all_errors)


# ---------------------------------------------------------------------------
# Collection loader
# ---------------------------------------------------------------------------

_COLLECTION_MERGE = """\
UNWIND $batch AS row
MERGE (n:Collection {id: row.id})
SET n.name_ar            = row.name_ar,
    n.name_en            = row.name_en,
    n.compiler_name      = row.compiler_name,
    n.compilation_year_ah = row.compilation_year_ah,
    n.sect               = row.sect,
    n.total_hadiths      = row.total_hadiths,
    n.expected_count     = row.expected_count,
    n.source_corpus      = row.source_corpus
"""


def _load_collections(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> LoadResult:
    """Load Collection nodes from collections_*.parquet files."""
    files = _filter_parquet_files(
        _parquet_files(staging_dir, "collections_"), staging_dir, skip_files
    )
    if not files:
        if strict:
            msg = f"No collections_*.parquet files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("collection_files_missing", dir=str(staging_dir))
        return LoadResult("Collection", 0, 0, 0)

    total_created = 0
    total_skipped = 0
    all_errors: list[str] = []
    total_batch = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        batch: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            cid = row.get("collection_id")
            if not cid or not isinstance(cid, str):
                all_errors.append(f"{fp.name} row {i}: invalid collection_id={cid!r}")
                total_skipped += 1
                continue
            # Canonical composition (da#191): drop Collection nodes for non-canonical
            # (source_corpus, collection) pairs so they don't outlive their Hadith.
            # The collection slug is the last ``:``-segment of collection_id.
            if not is_canonical_hadith(_val(row, "source_corpus", ""), cid.rsplit(":", 1)[-1]):
                total_skipped += 1
                continue
            full_id = collection_node_id(cid)
            batch.append(
                {
                    "id": full_id,
                    "name_ar": _val(row, "name_ar"),
                    "name_en": _val(row, "name_en", ""),
                    "compiler_name": _val(row, "compiler_name"),
                    "compilation_year_ah": _val(row, "compilation_year_ah"),
                    "sect": _val(row, "sect", ""),
                    "total_hadiths": _val(row, "total_hadiths"),
                    "expected_count": _val(row, "expected_count"),
                    "source_corpus": _val(row, "source_corpus", ""),
                }
            )
        if batch:
            total_created += client.execute_write_batch(_COLLECTION_MERGE, batch)
            total_batch += len(batch)

    merged = total_batch - total_created
    logger.info(
        "collections_loaded",
        files=len(files),
        created=total_created,
        merged=merged,
        skipped=total_skipped,
    )
    return LoadResult("Collection", total_created, merged, total_skipped, all_errors)


# ---------------------------------------------------------------------------
# Chain loader
# ---------------------------------------------------------------------------

_CHAIN_MERGE = """\
UNWIND $batch AS row
MERGE (n:Chain {id: row.id})
SET n.hadith_id           = row.hadith_id,
    n.chain_index         = row.chain_index,
    n.full_chain_text_ar  = row.full_chain_text_ar,
    n.full_chain_text_en  = row.full_chain_text_en,
    n.chain_length        = row.chain_length,
    n.is_complete         = row.is_complete,
    n.is_elevated         = row.is_elevated,
    n.classification      = row.classification,
    n.narrator_ids        = row.narrator_ids
"""


def _load_chains(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> LoadResult:
    """Load Chain nodes from the *resolved* narrator-mention master.

    Chains are synthesized from narrator-mention parquet: each unique
    (hadith_id, chain_index=0) tuple produces a Chain node whose ``narrator_ids``
    is the per-hadith mention sequence (ordered by ``position_in_chain``).

    Reads ``narrator_mentions_resolved*.parquet`` from ``curated_dir`` — NOT the
    raw ``narrator_mentions_*.parquet`` in ``staging_dir``. The canonical
    ``canonical_narrator_id`` column is produced by the resolve stage and written
    only to the curated master; the staging mentions have no such column, so
    reading staging produced hollow chains (every ``narrator_ids == []``,
    ``chain_length == 0``) — the #723 "chains empty" defect. ``staging_dir`` is
    retained only so ``skip_files`` paths resolve consistently with the other
    node loaders.
    """
    files = [
        fp
        for fp in _parquet_files(curated_dir, "narrator_mentions_resolved")
        if not _should_skip_file(fp, staging_dir, skip_files)
    ]
    if not files:
        if strict:
            msg = f"No narrator_mentions_resolved*.parquet files in {curated_dir}"
            raise FileNotFoundError(msg)
        logger.warning("chain_files_missing", dir=str(curated_dir))
        return LoadResult("Chain", 0, 0, 0)

    # Accumulate only the (position, canonical_id) pairs we need per hadith —
    # the resolved master is ~3.2M rows, so keeping whole row dicts would bloat
    # memory needlessly (see the streaming work in load_edges for the same reason).
    seen_hadiths: dict[str, list[tuple[int, str]]] = {}
    for fp in files:
        for row in _read_parquet_rows(fp):
            hid = row.get("source_hadith_id") or row.get("hadith_id")
            nid = row.get("canonical_narrator_id")
            if not hid or not nid:
                continue
            pos = row.get("position_in_chain") or 0
            seen_hadiths.setdefault(hid, []).append((pos, nid))

    batch: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0

    for hid, mentions in seen_hadiths.items():
        chn_id = chain_node_id(hid, 0)
        narrator_ids = [nid for _pos, nid in sorted(mentions, key=lambda pair: pair[0])]
        batch.append(
            {
                "id": chn_id,
                "hadith_id": hadith_node_id(hid),
                "chain_index": 0,
                "full_chain_text_ar": None,
                "full_chain_text_en": None,
                "chain_length": len(narrator_ids),
                "is_complete": len(narrator_ids) > 0,
                "is_elevated": False,
                "classification": "unknown",
                "narrator_ids": narrator_ids,
            }
        )

    created = client.execute_write_batch(_CHAIN_MERGE, batch) if batch else 0
    merged = len(batch) - created
    logger.info("chains_loaded", created=created, merged=merged, skipped=skipped)
    return LoadResult("Chain", created, merged, skipped, errors)


# ---------------------------------------------------------------------------
# Grading loader
# ---------------------------------------------------------------------------

_GRADING_MERGE = """\
UNWIND $batch AS row
MERGE (n:Grading {id: row.id})
SET n.hadith_id          = row.hadith_id,
    n.scholar_name       = row.scholar_name,
    n.grade              = row.grade,
    n.grade_normalized   = row.grade_normalized,
    n.methodology_school = row.methodology_school,
    n.era                = row.era
"""


def _load_gradings(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> LoadResult:
    """Load Grading nodes from hadith staging data.

    Gradings are extracted from the ``grade`` column of hadiths_*.parquet.
    Each hadith with a non-null grade produces a single Grading node
    attributed to the collection compiler.
    """
    files = _filter_parquet_files(_parquet_files(staging_dir, "hadiths_"), staging_dir, skip_files)
    if not files:
        if strict:
            msg = f"No hadiths_*.parquet files for grading extraction in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("grading_files_missing", dir=str(staging_dir))
        return LoadResult("Grading", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        for i, row in enumerate(rows):
            grade = row.get("grade")
            if not grade:
                continue
            sid = row.get("source_id")
            if not sid:
                errors.append(f"{fp.name} row {i}: grade present but no source_id")
                skipped += 1
                continue
            gid = grading_node_id(sid)
            batch.append(
                {
                    "id": gid,
                    "hadith_id": hadith_node_id(sid),
                    "scholar_name": _val(row, "collection_name", "unknown"),
                    "grade": grade,
                    # Normalized display grade (da#148): the raw ``grade`` may be
                    # Arabic / mixed-script free text; map it onto the HadithGrade
                    # vocabulary so the UI and queries have a stable key.
                    "grade_normalized": normalize_grade(grade),
                    "methodology_school": None,
                    "era": None,
                }
            )

    created = client.execute_write_batch(_GRADING_MERGE, batch) if batch else 0
    merged = len(batch) - created
    logger.info("gradings_loaded", created=created, merged=merged, skipped=skipped)
    return LoadResult("Grading", created, merged, skipped, errors)


# ---------------------------------------------------------------------------
# HistoricalEvent loader
# ---------------------------------------------------------------------------

_EVENT_MERGE = """\
UNWIND $batch AS row
MERGE (n:HistoricalEvent {id: row.id})
SET n.name_en       = row.name_en,
    n.name_ar       = row.name_ar,
    n.year_start_ah = row.year_start_ah,
    n.year_end_ah   = row.year_end_ah,
    n.year_start_ce = row.year_start_ce,
    n.year_end_ce   = row.year_end_ce,
    n.event_type    = row.event_type,
    n.caliphate     = row.caliphate,
    n.region        = row.region,
    n.description   = row.description,
    n.source_url    = row.source_url
"""


def _load_historical_events(
    client: Neo4jClient,
    curated_dir: Path,
    *,
    strict: bool = True,
) -> LoadResult:
    """Load HistoricalEvent nodes from historical_events.yaml."""
    path = curated_dir / "historical_events.yaml"
    if not path.exists():
        if strict:
            msg = f"Missing required file: {path}"
            raise FileNotFoundError(msg)
        logger.warning("historical_events_missing", path=str(path))
        return LoadResult("HistoricalEvent", 0, 0, 0)

    with open(path) as f:
        data = yaml.safe_load(f)

    events = data.get("events", [])
    batch: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0

    for i, evt in enumerate(events):
        eid = evt.get("id")
        if not eid or not isinstance(eid, str):
            errors.append(f"event {i}: invalid id={eid!r}")
            skipped += 1
            continue
        name_en = evt.get("name_en")
        if not name_en:
            errors.append(f"event {i}: missing name_en")
            skipped += 1
            continue
        batch.append(
            {
                "id": eid,
                "name_en": name_en,
                "name_ar": evt.get("name_ar"),
                "year_start_ah": evt.get("year_start_ah"),
                "year_end_ah": evt.get("year_end_ah"),
                "year_start_ce": evt.get("year_start_ce"),
                "year_end_ce": evt.get("year_end_ce"),
                "event_type": evt.get("type"),
                "caliphate": evt.get("caliphate"),
                "region": evt.get("region"),
                "description": evt.get("description"),
                "source_url": evt.get("source_url"),
            }
        )

    created = client.execute_write_batch(_EVENT_MERGE, batch) if batch else 0
    merged = len(batch) - created
    logger.info("historical_events_loaded", created=created, merged=merged, skipped=skipped)
    return LoadResult("HistoricalEvent", created, merged, skipped, errors)


# ---------------------------------------------------------------------------
# Location loader
# ---------------------------------------------------------------------------

_LOCATION_MERGE = """\
UNWIND $batch AS row
MERGE (n:Location {id: row.id})
SET n.name_en  = row.name_en,
    n.name_ar  = row.name_ar,
    n.region   = row.region,
    n.lat      = row.lat,
    n.lon      = row.lon
"""


def _load_locations(
    client: Neo4jClient,
    curated_dir: Path,
    *,
    strict: bool = True,
) -> LoadResult:
    """Load Location nodes from locations.yaml if available."""
    path = curated_dir / "locations.yaml"
    if not path.exists():
        if strict:
            logger.warning("locations_file_missing", path=str(path))
        return LoadResult("Location", 0, 0, 0)

    with open(path) as f:
        data = yaml.safe_load(f)

    locations = data.get("locations", [])
    batch: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0

    for i, loc in enumerate(locations):
        lid = loc.get("id")
        if not lid or not isinstance(lid, str):
            errors.append(f"location {i}: invalid id={lid!r}")
            skipped += 1
            continue
        full_id = f"loc:{lid}" if not lid.startswith("loc:") else lid
        batch.append(
            {
                "id": full_id,
                "name_en": loc.get("name_en", ""),
                "name_ar": loc.get("name_ar"),
                "region": loc.get("region"),
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
            }
        )

    created = client.execute_write_batch(_LOCATION_MERGE, batch) if batch else 0
    merged = len(batch) - created
    logger.info("locations_loaded", created=created, merged=merged, skipped=skipped)
    return LoadResult("Location", created, merged, skipped, errors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _should_skip_file(path: Path, staging_dir: Path, skip_files: list[str] | None) -> bool:
    """Check if a file should be skipped based on the skip_files list."""
    if not skip_files:
        return False
    # Build a key like "staging/hadiths_bukhari.parquet"
    try:
        rel = path.relative_to(staging_dir.parent)
        return str(rel) in skip_files
    except ValueError:
        return False


def _filter_parquet_files(
    files: list[Path], staging_dir: Path, skip_files: list[str] | None
) -> list[Path]:
    """Filter out files that are in the skip_files list."""
    if not skip_files:
        return files
    return [f for f in files if not _should_skip_file(f, staging_dir, skip_files)]


def load_all_nodes(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,
    *,
    strict: bool = True,
    skip_files: list[str] | None = None,
) -> list[LoadResult]:
    """Load all node types into Neo4j.

    Ensures uniqueness constraints first, then loads each node type
    in dependency order (narrators before chains, hadiths before gradings).

    Parameters
    ----------
    skip_files:
        Manifest keys of files to skip (for incremental loading).
    """
    client.ensure_constraints()
    client.ensure_fulltext_indexes()

    results: list[LoadResult] = []
    results.append(
        _load_narrators(client, staging_dir, curated_dir, strict=strict, skip_files=skip_files)
    )
    results.append(_load_hadiths(client, staging_dir, strict=strict, skip_files=skip_files))
    results.append(_load_collections(client, staging_dir, strict=strict, skip_files=skip_files))
    results.append(
        _load_chains(client, staging_dir, curated_dir, strict=strict, skip_files=skip_files)
    )
    results.append(_load_gradings(client, staging_dir, strict=strict, skip_files=skip_files))
    results.append(_load_historical_events(client, curated_dir, strict=strict))
    results.append(_load_locations(client, curated_dir, strict=strict))

    total_created = sum(r.created for r in results)
    total_errors = sum(len(r.validation_errors) for r in results)
    logger.info(
        "all_nodes_loaded",
        total_created=total_created,
        total_errors=total_errors,
        node_types=len(results),
    )
    return results
