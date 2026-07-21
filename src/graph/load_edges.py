"""Neo4j edge/relationship loading.

Batch UNWIND loaders for 6 relationship types: TRANSMITTED_TO, NARRATED,
APPEARS_IN, PARALLEL_OF, STUDIED_UNDER, and GRADED_BY.  Each loader uses
MATCH (not MERGE) for endpoints, logging and counting missing endpoints
rather than silently creating dangling references.  Edge creation uses MERGE
for idempotent re-runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.graph.load_nodes import build_loaded_hadith_ids
from src.parse.composition import is_canonical_hadith_id
from src.parse.identity import (
    DoubledCorpusPrefixError,
    collection_node_id,
    grading_node_id,
    hadith_node_id,
    make_canonical_id,
    narrator_node_id,
)
from src.parse.schemas import EDGE_RELATION_STUDIED_UNDER, VALID_EDGE_RELATIONS
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = ["load_all_edges", "EdgeLoadResult"]

DEFAULT_BATCH_SIZE = 1000


@dataclass(frozen=True)
class EdgeLoadResult:
    """Outcome of loading a single edge/relationship type."""

    edge_type: str
    created: int
    skipped: int
    missing_endpoints: int
    # Edges refused because an endpoint id violates the id grammar (da#359).
    malformed_ids: int = 0

    @property
    def refused(self) -> int:
        """Edges declined because the input was defective, by any cause.

        The edge loaders' only input-defect refusal is a malformed endpoint id;
        a missing endpoint is counted separately as ``missing_endpoints``,
        because a Hadith node that was never loaded is a *consequence* of a
        refusal upstream rather than a defect in the edge row itself.
        """
        return self.malformed_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked_read(
    client: Neo4jClient,
    query: str,
    batch: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Execute a read query in chunks to avoid memory issues at scale."""
    results: list[dict[str, Any]] = []
    for i in range(0, len(batch), batch_size):
        chunk = batch[i : i + batch_size]
        results.extend(client.execute_read(query, {"batch": chunk}))
    return results


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file and return row dicts (whole-file, in memory).

    Fine for the small mention/network/hadith files. For multi-million-row files
    (``parallel_links.parquet`` — 6.66M rows at the #723 reload) use
    :func:`_iter_parquet_row_batches` instead: materializing the whole table via
    ``to_pylist()`` plus a second transformed copy peaked the loader past the
    host's RAM and the container was OOM-killed (137, #723 stage load).
    """
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


# Row-batch size for streaming large edge inputs. Independent of the Neo4j write
# ``batch_size`` (1000): this bounds how many parquet rows are materialized as
# Python dicts at once, so loader peak memory is ~constant regardless of file
# size. 50k rows ≈ tens of MB, well under the host's per-process headroom.
_READ_BATCH_ROWS = 50_000


def _iter_parquet_row_batches(
    path: Path, batch_rows: int = _READ_BATCH_ROWS
) -> Iterator[list[dict[str, Any]]]:
    """Yield row-dict batches from a Parquet file without materializing it whole.

    The streaming counterpart to :func:`_read_parquet_rows`. Lets a loader process
    millions of rows (transform → endpoint-check → write → discard per batch) while
    holding only ``batch_rows`` rows at a time — the fix for the #723 PARALLEL_OF
    OOM, where the 6.66M-row ``parallel_links.parquet`` was read whole.
    """
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_rows):
        yield record_batch.to_pylist()


def _parquet_files(directory: Path, prefix: str) -> list[Path]:
    """Return sorted parquet files matching *prefix* in *directory*."""
    return sorted(directory.glob(f"{prefix}*.parquet"))


def _val(row: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from *row*, returning *default* for ``None``."""
    v = row.get(key)
    return default if v is None else v


def _chain_index(row: dict[str, Any]) -> int:
    """Per-hadith isnad-chain ordinal for a mention row (da#282).

    Chain edges are grouped by ``(hadith_id, chain_index)`` — never by hadith
    alone — so a hadith carrying more than one transmission sequence (today: lk's
    Arabic + English isnads, each numbered from position 0) does not have its
    chains flattened into one position-sorted list, which fabricated narrator
    adjacencies that exist in no actual chain. A missing/None ``chain_index``
    (legacy pre-da#282 file, or a single-isnad producer that left it unset)
    coalesces to 0, so single-chain hadiths are unaffected.
    """
    v = row.get("chain_index")
    return int(v) if v is not None else 0


def _staging_loaded_hadith_ids(staging_dir: Path) -> set[str] | None:
    """The ``Hadith.id`` node ids the node loader loads from *staging_dir* (da#373).

    Builds the set from the SAME ``hadiths_*.parquet`` the node loader reads, via
    :func:`src.graph.load_nodes.build_loaded_hadith_ids`, so the chain-edge gate
    sees the node loader's real kept set — composition AND cross-edition matn
    dedup — not just the id-grammar composition slice ``is_canonical_hadith_id``
    can re-derive. Returns ``None`` when *staging_dir* holds no hadith files, in
    which case the caller falls back to the id-grammar gate (pre-da#373 behavior)
    — there are no Hadith nodes to orphan against anyway.
    """
    files = _parquet_files(staging_dir, "hadiths_")
    if not files:
        return None
    return build_loaded_hadith_ids(files)


def _mention_hadith_survives(hid: str, loaded_hadith_ids: set[str] | None) -> bool:
    """True if a mention's hadith should still yield a chain edge (da#333 + da#373).

    Drops exactly the mentions whose hadith the node loader did NOT load, so a
    TRANSMITTED_TO / NARRATED edge is never emitted against a missing Hadith node:

    * ``loaded_hadith_ids`` is the node loader's real kept set (da#373). A
      well-formed id absent from it — a non-canonical edition (da#333, the ~196k
      fawaz six-books chains) OR a cross-edition-deduped sanadset tradition — is
      dropped. The cross-edition half is the one ``is_canonical_hadith_id``
      structurally could not see: the matn dedup decision is not recoverable from
      the id string.
    * ``None`` means no staging hadith files were available to consult, so fall
      back to the id-grammar composition gate (:func:`is_canonical_hadith_id`).

    **Total** — a malformed id returns ``True`` (survives here) and is deferred to
    the per-group malformed quarantine that counts it (da#355); it is never
    silently reclassified as a non-canonical drop, mirroring the totality of
    ``is_canonical_hadith_id``.
    """
    if loaded_hadith_ids is None:
        return is_canonical_hadith_id(hid)
    try:
        node_id = hadith_node_id(hid)
    except DoubledCorpusPrefixError:
        return True  # defer to the da#355 malformed quarantine
    return node_id in loaded_hadith_ids


# ---------------------------------------------------------------------------
# 1. TRANSMITTED_TO — consecutive narrator pairs in each chain
# ---------------------------------------------------------------------------

_TRANSMITTED_TO_QUERY = """\
UNWIND $batch AS row
MATCH (n1:Narrator {id: row.from_id})
MATCH (n2:Narrator {id: row.to_id})
MERGE (n1)-[:TRANSMITTED_TO {
    position_in_chain: row.position,
    hadith_id: row.hadith_id
}]->(n2)
"""

_TRANSMITTED_TO_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (n1:Narrator {id: row.from_id})
OPTIONAL MATCH (n2:Narrator {id: row.to_id})
RETURN row.from_id AS from_id,
       row.to_id AS to_id,
       n1 IS NOT NULL AS from_exists,
       n2 IS NOT NULL AS to_exists
"""


def _build_chain_pairs(
    mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build consecutive narrator pairs from sorted chain mentions.

    Each mention must have ``canonical_narrator_id``, ``position_in_chain``,
    and ``hadith_id``.  Returns dicts with ``from_id``, ``to_id``,
    ``position``, and ``hadith_id``.

    The emitted ``hadith_id`` is canonicalized through :func:`hadith_node_id`
    (adds the ``hdt:`` prefix and collapses the main#139 double-prefix) so it
    matches ``Hadith.id`` — the value ``GET /graph/hadith/{id}/chain`` joins on.
    Storing the raw staging id here (``sanadset:sanadset:...``) was da#325: it
    matched no Hadith node, so the chain reconstruction returned empty for every
    hadith. A missing/empty raw id is preserved as ``""`` (never a bare ``hdt:``).
    """
    sorted_mentions = sorted(mentions, key=lambda r: r.get("position_in_chain", 0))
    # Filter to mentions with resolved narrator IDs
    resolved = [m for m in sorted_mentions if m.get("canonical_narrator_id")]
    pairs: list[dict[str, Any]] = []
    for i in range(len(resolved) - 1):
        from_id = resolved[i]["canonical_narrator_id"]
        to_id = resolved[i + 1]["canonical_narrator_id"]
        # Canonicalize the hadith id to match Hadith.id (da#325). Guard the empty
        # case: a missing id stays "" rather than becoming a bare "hdt:".
        raw_hadith_id = resolved[i].get("hadith_id", "")
        hadith_id = hadith_node_id(raw_hadith_id) if raw_hadith_id else ""
        # Drop self-loops: two ADJACENT mentions that resolve to the same
        # canonical narrator (a repeated narrator in the raw chain, or two
        # mentions the disambiguator collapsed onto one node) would otherwise
        # MERGE a TRANSMITTED_TO edge from a narrator to itself (da#148 — 8 live
        # self-loops). The network-edge producers already guard this
        # (``muhaddithat._parse_network_edges`` skips ``from_id == to_id``,
        # ``mis`` skips ``from_name == to_name``); applying it on the
        # chain-derived path keeps a re-load self-loop-free everywhere.
        if from_id == to_id:
            logger.debug(
                "transmitted_to_self_loop_skipped",
                narrator_id=from_id,
                hadith_id=hadith_id,
            )
            continue
        pairs.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "position": resolved[i].get("position_in_chain", i),
                "hadith_id": hadith_id,
            }
        )
    return pairs


def _resolved_mentions_files(staging_dir: Path, curated_dir: Path | None) -> list[Path]:
    """Locate the mention files the chain edges (NARRATED / TRANSMITTED_TO) read.

    The chain edges key on ``canonical_narrator_id`` — and the only mentions that
    carry it are the RESOLVED ones, which ``resolve.run_all`` writes to the
    **curated** (output) dir as ``narrator_mentions_resolved.parquet``, NOT to
    staging (``src/resolve/ner.py`` / ``disambiguate.py``). Prefer that curated
    file; fall back to a resolved file that happens to sit in staging, then to
    the raw per-source ``narrator_mentions_*`` (pre-resolve — no canonical id, so
    it yields no edges, but it keeps strict-mode file-presence checks satisfied).

    Wiring curated_dir here is what fixes the orchestrated load emitting zero
    NARRATED/TRANSMITTED_TO edges on real data (da#141 / main#601 criterion #1):
    before this, both loaders globbed staging only and silently fell back to the
    raw mentions.
    """
    if curated_dir is not None:
        files = _parquet_files(curated_dir, "narrator_mentions_resolved")
        if files:
            return files
    files = _parquet_files(staging_dir, "narrator_mentions_resolved")
    if files:
        return files
    return _parquet_files(staging_dir, "narrator_mentions_")


def _load_transmitted_to(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    curated_dir: Path | None = None,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    loaded_hadith_ids: set[str] | None = None,
) -> EdgeLoadResult:
    """Load TRANSMITTED_TO edges from narrator_mentions_resolved.parquet.

    *loaded_hadith_ids* is the node loader's real kept set (da#373); when the
    caller does not supply it, it is derived from *staging_dir*'s hadith files.
    See :func:`_mention_hadith_survives`.
    """
    if loaded_hadith_ids is None:
        loaded_hadith_ids = _staging_loaded_hadith_ids(staging_dir)
    files = _resolved_mentions_files(staging_dir, curated_dir)
    if not files:
        if strict:
            msg = f"No narrator_mentions files in {curated_dir or staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("transmitted_to_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("TRANSMITTED_TO", 0, 0, 0)

    # Group mentions by (hadith_id, chain_index) so each isnad-chain is sequenced
    # on its own (da#282). Keying on hadith alone flattened every chain of a
    # multi-isnad hadith into one position-sorted list — see ``_chain_index``.
    #
    # Curated mention-links (provenance-bearing rows from the muhaddithat orphan
    # producer, da#228) are NARRATED-ONLY by contract and MUST NOT enter chain-pair
    # construction. _resolved_mentions_files is a PREFIX glob, so the curated
    # ``narrator_mentions_resolved_muhaddithat.parquet`` is read here alongside the
    # main resolved-mentions file; a curated link's ``hadith_id`` is a real
    # ``sunnah`` hadith id, which that hadith's own NER chain mention also carries
    # (same id grammar). Without this guard, _build_chain_pairs would pair the
    # orphan narrator with that hadith's Companion narrator into a FABRICATED
    # TRANSMITTED_TO edge — a wrong attribution ADR-004 forbids, which the
    # narrator-only _TRANSMITTED_TO_CHECK cannot catch (both endpoints are real
    # nodes). Dropping provenance-bearing rows here keeps them flowing to NARRATED
    # (the correct edge) while never fabricating a transmission pair. A normal
    # chain mention has no ``provenance`` column (row.get → None), so this only
    # ever excludes curated links.
    by_chain: dict[tuple[str, int], list[dict[str, Any]]] = {}
    dropped_noncanonical = 0
    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            hid = row.get("hadith_id") or row.get("source_hadith_id")
            if not hid:
                continue
            if not _mention_hadith_survives(hid, loaded_hadith_ids):
                # da#333: a mention whose hadith is NOT a canonical node — e.g.
                # fawaz's six-books chains, deduplicated to the lk spine — must not
                # produce a TRANSMITTED_TO edge, mirroring the node dedup
                # (``load_nodes._load_hadiths``). Without this gate the edge orphans
                # against a Hadith node that was never loaded (the ~196k
                # ``fawaz:<book>:<n>`` dangling chains that duplicate lk). ``mis`` is
                # unaffected here for a different reason: it produces no narrator
                # mentions at all (absent from NER routing), so it never reaches this
                # narrator-mention path. Its transmission pairs are staged as
                # ``network_edges_mis.parquet`` and ARE read by the STUDIED_UNDER glob
                # (:func:`_load_studied_under`), but skipped row-by-row because they
                # declare TRANSMITTED_TO, not STUDIED_UNDER — so no mis edge is
                # persisted (produced, not loaded as a graph edge), pending a
                # name-reconciliation crosswalk (mis Latin-transliteration ids ↔
                # Arabic-normalized canonical keys); tracked as a separate follow-up
                # to da#364.
                dropped_noncanonical += 1
                continue
            if row.get("provenance"):
                continue  # curated NARRATED-only link — never a transmission pair
            by_chain.setdefault((hid, _chain_index(row)), []).append(row)

    # Build all chain pairs. Validate the group's hadith id ONCE up front: every
    # pair _build_chain_pairs emits re-derives it, so a malformed id would raise
    # from inside the builder. Quarantine the whole group instead (da#355) — this
    # loader commits per batch inside the streaming loop, so an escaping exception
    # leaves batches 1..N-1 already written to Neo4j.
    all_pairs: list[dict[str, Any]] = []
    malformed_ids = 0
    for (hid, _cidx), mentions in by_chain.items():
        try:
            hadith_node_id(hid)
        except DoubledCorpusPrefixError:
            if strict:
                raise
            malformed_ids += 1
            continue
        all_pairs.extend(_build_chain_pairs(mentions))
    if malformed_ids:
        logger.warning("transmitted_to_malformed_ids_quarantined", hadiths=malformed_ids)

    if not all_pairs:
        logger.info("transmitted_to_no_pairs", dropped_noncanonical=dropped_noncanonical)
        return EdgeLoadResult("TRANSMITTED_TO", 0, 0, 0, malformed_ids=malformed_ids)

    # Check for missing endpoints
    check_results = _chunked_read(client, _TRANSMITTED_TO_CHECK, all_pairs, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for pair, check in zip(all_pairs, check_results):
        if check.get("from_exists") and check.get("to_exists"):
            valid_batch.append(pair)
        else:
            missing += 1
            if not check.get("from_exists"):
                logger.debug("transmitted_to_missing_from", id=pair["from_id"])
            if not check.get("to_exists"):
                logger.debug("transmitted_to_missing_to", id=pair["to_id"])

    created = (
        client.execute_write_batch(_TRANSMITTED_TO_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info(
        "transmitted_to_loaded",
        created=created,
        missing_endpoints=missing,
        total_pairs=len(all_pairs),
        dropped_noncanonical=dropped_noncanonical,
    )
    return EdgeLoadResult("TRANSMITTED_TO", created, 0, missing, malformed_ids=malformed_ids)


# ---------------------------------------------------------------------------
# 2. NARRATED — first narrator in each chain -> hadith
# ---------------------------------------------------------------------------

# Identity of a NARRATED edge is the (narrator, hadith) PAIR. ``provenance`` — the
# source that attests this narrator transmitted this hadith — is an *attribute* of
# that one edge, not part of its key, so it is SET after a property-less MERGE
# (never inside the MERGE pattern: Neo4j refuses to MERGE a relationship with a
# null property, and an ordinary chain-derived NARRATED carries no provenance).
# The coalesce-preserve form mirrors the APPEARS_IN / PARALLEL_OF contract (da#77):
# a row with a null provenance keeps any value already on the edge rather than
# clearing it, so a re-load is idempotent and null-safe. The curated muhaddithat
# orphan-links (da#228, ``src/resolve/muhaddithat_links.py``) are the producer that
# populates it; chain-derived mentions leave it null (the property stays absent).
_NARRATED_QUERY = """\
UNWIND $batch AS row
MATCH (n:Narrator {id: row.narrator_id})
MATCH (h:Hadith {id: row.hadith_id})
MERGE (n)-[r:NARRATED]->(h)
SET r.provenance = coalesce(row.provenance, r.provenance)
"""

_NARRATED_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (n:Narrator {id: row.narrator_id})
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
RETURN row.narrator_id AS narrator_id,
       row.hadith_id AS hadith_id,
       n IS NOT NULL AS narrator_exists,
       h IS NOT NULL AS hadith_exists
"""


def _load_narrated(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    curated_dir: Path | None = None,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    loaded_hadith_ids: set[str] | None = None,
) -> EdgeLoadResult:
    """Load NARRATED edges — first narrator (position 0) in each chain -> hadith.

    *loaded_hadith_ids* is the node loader's real kept set (da#373); derived from
    *staging_dir* when not supplied. See :func:`_mention_hadith_survives`.
    """
    if loaded_hadith_ids is None:
        loaded_hadith_ids = _staging_loaded_hadith_ids(staging_dir)
    files = _resolved_mentions_files(staging_dir, curated_dir)
    if not files:
        if strict:
            msg = f"No narrator_mentions files in {curated_dir or staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("narrated_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("NARRATED", 0, 0, 0)

    # Find the position-0 narrator of EACH chain (lowest position_in_chain within
    # a (hadith_id, chain_index) group), not per hadith (da#282). A multi-isnad
    # hadith is narrated by the first narrator of every one of its chains; keying on
    # hadith alone dropped all but one chain's opener. Distinct chains that share a
    # first narrator (e.g. lk's Arabic + English isnads) still MERGE to one edge, so
    # this never inflates NARRATED. The winning mention's ``provenance`` (present
    # only on curated orphan-links, da#228; null for ordinary chain mentions) rides
    # along so it lands on the NARRATED edge.
    # (hadith_id, chain_index) -> (pos, nid, provenance)
    first_narrators: dict[tuple[str, int], tuple[int, str, str | None]] = {}
    dropped_noncanonical = 0
    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            hid = row.get("hadith_id") or row.get("source_hadith_id")
            nid = row.get("canonical_narrator_id")
            pos = row.get("position_in_chain", 0)
            if not hid or not nid:
                continue
            if not _mention_hadith_survives(hid, loaded_hadith_ids):
                # da#333: mirror the node dedup — a mention whose hadith is not a
                # canonical node (fawaz six-books, deduped to lk) yields no NARRATED
                # edge. Such an edge would otherwise be a guaranteed missing-endpoint
                # (no Hadith node exists for it); dropping it at the source keeps the
                # edge path consistent with the node loader. ``mis`` never reaches
                # here because it produces no narrator mentions; its transmission
                # pairs are staged as ``network_edges_mis.parquet`` and read by the
                # STUDIED_UNDER glob but skipped row-by-row (they declare
                # TRANSMITTED_TO, not STUDIED_UNDER), so no mis edge is persisted
                # (produced, not loaded as a graph edge), pending a
                # name-reconciliation crosswalk; tracked as a separate follow-up to
                # da#364.
                dropped_noncanonical += 1
                continue
            key = (hid, _chain_index(row))
            if key not in first_narrators or pos < first_narrators[key][0]:
                first_narrators[key] = (pos, nid, row.get("provenance"))

    batch: list[dict[str, Any]] = []
    malformed_ids = 0
    for (hid, _cidx), (_pos, nid, provenance) in first_narrators.items():
        try:
            full_hid = hadith_node_id(hid)
        except DoubledCorpusPrefixError:
            if strict:
                raise
            malformed_ids += 1
            continue
        batch.append({"narrator_id": nid, "hadith_id": full_hid, "provenance": provenance})
    if malformed_ids:
        logger.warning("narrated_malformed_ids_quarantined", hadiths=malformed_ids)

    if not batch:
        logger.info("narrated_no_edges")
        return EdgeLoadResult("NARRATED", 0, 0, 0, malformed_ids=malformed_ids)

    # Check endpoints
    check_results = _chunked_read(client, _NARRATED_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("narrator_exists") and check.get("hadith_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_NARRATED_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info(
        "narrated_loaded",
        created=created,
        missing_endpoints=missing,
        dropped_noncanonical=dropped_noncanonical,
    )
    return EdgeLoadResult("NARRATED", created, 0, missing, malformed_ids=malformed_ids)


# ---------------------------------------------------------------------------
# 3. APPEARS_IN — hadith -> collection
# ---------------------------------------------------------------------------

# Identity of an APPEARS_IN edge is the (hadith, collection) PAIR — a hadith
# appears in a given collection at exactly one position, so the positional
# values (book / chapter / hadith number) are *attributes* of that single edge,
# not part of its key. They are therefore SET after the MERGE, never inside the
# MERGE pattern (da#77): Neo4j refuses to MERGE a relationship with a null
# property value, and the scraper leaves ``hadith_number`` null until da#72, so a
# keyed-on-properties MERGE aborts the whole load on real scraped data.
#
# The SET uses the streaming ingest path's coalesce-preserve contract
# (ingest-platform ``workers/ingest/processor.py _build_edge_cypher``:
# ``r.<f> = coalesce(row.<f>, r.<f>)``) so the two ingest paths converge byte-for-
# byte on idempotency: a row with an explicit-null positional value preserves any
# value already on the edge rather than clearing it. This keeps dedup correct
# (one APPEARS_IN per hadith->collection) and is null-safe.
_APPEARS_IN_QUERY = """\
UNWIND $batch AS row
MATCH (h:Hadith {id: row.hadith_id})
MATCH (c:Collection {id: row.collection_id})
MERGE (h)-[r:APPEARS_IN]->(c)
SET r.book_number = coalesce(row.book_number, r.book_number),
    r.chapter_number = coalesce(row.chapter_number, r.chapter_number),
    r.hadith_number_in_book = coalesce(row.hadith_number, r.hadith_number_in_book)
"""

_APPEARS_IN_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
OPTIONAL MATCH (c:Collection {id: row.collection_id})
RETURN row.hadith_id AS hadith_id,
       row.collection_id AS collection_id,
       h IS NOT NULL AS hadith_exists,
       c IS NOT NULL AS collection_exists
"""


def _load_appears_in(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load APPEARS_IN edges — hadith -> collection with positional properties."""
    files = _parquet_files(staging_dir, "hadiths_")
    if not files:
        if strict:
            msg = f"No hadiths_*.parquet files in {staging_dir}"
            raise FileNotFoundError(msg)
        logger.warning("appears_in_files_missing", dir=str(staging_dir))
        return EdgeLoadResult("APPEARS_IN", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    skipped = 0
    malformed_ids = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            sid = row.get("source_id")
            cname = row.get("collection_name")
            if not sid or not cname:
                skipped += 1
                continue
            try:
                hid = hadith_node_id(sid)
            except DoubledCorpusPrefixError:
                if strict:
                    raise
                skipped += 1
                malformed_ids += 1
                continue
            # Collection IDs in staging use "{corpus}:{name}" format (e.g. "lk:bukhari").
            # Build the same key so we match the Collection nodes that were loaded.
            corpus = row.get("source_corpus", "")
            raw_cid = f"{corpus}:{cname}" if corpus else cname
            cid = collection_node_id(raw_cid)
            batch.append(
                {
                    "hadith_id": hid,
                    "collection_id": cid,
                    "book_number": _val(row, "book_number"),
                    "chapter_number": _val(row, "chapter_number"),
                    "hadith_number": _val(row, "hadith_number"),
                }
            )

    if malformed_ids:
        logger.warning("appears_in_malformed_ids_quarantined", rows=malformed_ids)

    if not batch:
        logger.info("appears_in_no_edges")
        return EdgeLoadResult("APPEARS_IN", 0, skipped, 0, malformed_ids=malformed_ids)

    # Check endpoints
    check_results = _chunked_read(client, _APPEARS_IN_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("hadith_exists") and check.get("collection_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_APPEARS_IN_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("appears_in_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("APPEARS_IN", created, skipped, missing, malformed_ids=malformed_ids)


# ---------------------------------------------------------------------------
# 4. PARALLEL_OF — from parallel_links.parquet
# ---------------------------------------------------------------------------

# Identity of a PARALLEL_OF edge is the (hadith_a, hadith_b) PAIR — the similarity
# score / variant tier / cross_sect flag are *attributes* of that one edge, not
# part of its key. They are therefore SET after a bare-edge MERGE, never inside
# the MERGE pattern (da#77 / main#139): Neo4j refuses to MERGE a relationship with
# a null property in the pattern, and keying on the properties would mint a SECOND
# edge between the same pair whenever a re-run produced a slightly different score
# (dedup is re-run as more corpora land). coalesce-preserve keeps re-runs
# idempotent and null-safe — exactly the APPEARS_IN contract.
_PARALLEL_OF_QUERY = """\
UNWIND $batch AS row
MATCH (h1:Hadith {id: row.id_a})
MATCH (h2:Hadith {id: row.id_b})
MERGE (h1)-[r:PARALLEL_OF]->(h2)
SET r.similarity_score = coalesce(row.score, r.similarity_score),
    r.variant_type = coalesce(row.variant_type, r.variant_type),
    r.cross_sect = coalesce(row.cross_sect, r.cross_sect)
"""

_PARALLEL_OF_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h1:Hadith {id: row.id_a})
OPTIONAL MATCH (h2:Hadith {id: row.id_b})
RETURN row.id_a AS id_a,
       row.id_b AS id_b,
       h1 IS NOT NULL AS a_exists,
       h2 IS NOT NULL AS b_exists
"""


def _load_parallel_of(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load PARALLEL_OF edges from parallel_links.parquet."""
    path = staging_dir / "parallel_links.parquet"
    if not path.exists():
        if strict:
            msg = f"Missing required file: {path}"
            raise FileNotFoundError(msg)
        logger.warning("parallel_links_missing", path=str(path))
        return EdgeLoadResult("PARALLEL_OF", 0, 0, 0)

    # Stream parquet row-batches so loader memory stays bounded regardless of file
    # size: the #723 reload's parallel_links.parquet has 6.66M rows, and reading it
    # whole (plus the transformed copy) OOM-killed the loader (137). Each batch is
    # transformed → endpoint-checked → written → discarded; totals accumulate across
    # batches so the da#160 silent-failure warnings below still key on file totals.
    candidates = 0
    skipped = 0
    missing = 0
    created = 0
    malformed_ids = 0
    for rows in _iter_parquet_row_batches(path):
        batch: list[dict[str, Any]] = []
        for row in rows:
            id_a = row.get("hadith_id_a")
            id_b = row.get("hadith_id_b")
            if not id_a or not id_b:
                skipped += 1
                continue
            # da#355: this loop commits each batch via execute_write_batch BEFORE
            # reading the next one — there is no spanning transaction. An exception
            # escaping here would leave batches 1..N-1 in Neo4j and abort with a
            # traceback, hours into the load. Quarantine the pair instead.
            try:
                full_a = hadith_node_id(id_a)
                full_b = hadith_node_id(id_b)
            except DoubledCorpusPrefixError:
                if strict:
                    raise
                skipped += 1
                malformed_ids += 1
                continue
            if full_a > full_b:
                full_a, full_b = full_b, full_a
            batch.append(
                {
                    "id_a": full_a,
                    "id_b": full_b,
                    "score": _val(row, "similarity_score", 0.0),
                    "variant_type": _val(row, "variant_type", "unknown"),
                    "cross_sect": _val(row, "cross_sect", False),
                }
            )

        if not batch:
            continue
        candidates += len(batch)

        # Check endpoints for this batch only
        check_results = _chunked_read(client, _PARALLEL_OF_CHECK, batch, batch_size)
        valid_batch: list[dict[str, Any]] = []
        for item, check in zip(batch, check_results):
            if check.get("a_exists") and check.get("b_exists"):
                valid_batch.append(item)
            else:
                missing += 1

        if valid_batch:
            created += client.execute_write_batch(
                _PARALLEL_OF_QUERY, valid_batch, batch_size=batch_size
            )

    if malformed_ids:
        logger.warning("parallel_of_malformed_ids_quarantined", pairs=malformed_ids)

    if candidates == 0:
        # An empty parallel_links.parquet means the dedup / parallel-detection
        # stage produced nothing — /compare Browse Parallels will be empty. Surface
        # it as a WARNING rather than a silent INFO (da#160 conformance).
        logger.warning(
            "parallel_of_no_edges",
            msg="parallel_links.parquet has no usable rows — /compare will be empty",
        )
        return EdgeLoadResult("PARALLEL_OF", 0, skipped, 0, malformed_ids=malformed_ids)

    # Detected links that resolved to zero loaded edges is a silent-failure mode
    # (every endpoint missing — e.g. an id-scheme mismatch): warn, do not pass it
    # by at INFO (da#160).
    if created == 0:
        logger.warning(
            "parallel_of_loaded_zero",
            candidates=candidates,
            skipped=skipped,
            missing_endpoints=missing,
            msg="parallel links present but 0 PARALLEL_OF edges loaded",
        )
    else:
        logger.info(
            "parallel_of_loaded", created=created, skipped=skipped, missing_endpoints=missing
        )
    return EdgeLoadResult("PARALLEL_OF", created, skipped, missing, malformed_ids=malformed_ids)


# ---------------------------------------------------------------------------
# 5. STUDIED_UNDER — from the studentship network_edges_*.parquet (muhaddithat, itqan)
# ---------------------------------------------------------------------------

# NETWORK_EDGE_SCHEMA is reused by producers whose edges mean DIFFERENT relations.
# muhaddithat + itqan emit teacher↔student (studentship) pairs that ARE
# STUDIED_UNDER. Others — e.g. `mis`, whose rows are *isnad transmission* pairs (a
# different relation type AND the opposite direction) — must NOT load as
# STUDIED_UNDER, or their edges would land as wrong-type, wrong-direction edges.
#
# Routing is by the per-row ``relation`` field on NETWORK_EDGE_SCHEMA (da#133): a
# row is loaded here iff it DECLARES STUDIED_UNDER. da#157 removes the old filename
# allowlist that silently defaulted a relation-LESS row to STUDIED_UNDER — that
# default was a footgun: the next isnad-transmission producer that forgot to set
# ``relation`` would have its edges mis-routed onto the studentship relation. The
# loader now REFUSES (raises on) any row that declares no — or an unrecognized —
# relation (:func:`_row_relation`). Every producer must set ``relation`` explicitly.


def _row_relation(row: dict[str, Any], *, source: str) -> str:
    """Relation a NETWORK_EDGE row declares — REQUIRED (da#133 / da#157).

    Returns the row's explicit ``relation``. Refuses a row that declares none, or
    one whose value is not a recognized relation, by raising ``ValueError`` —
    there is no default. This is the da#157 fail-safe: a relation-less edge is a
    producer defect that must never be silently routed onto the STUDIED_UNDER
    (studentship) allowlist. *source* names the offending file for the error.
    """
    rel = row.get("relation")
    if not isinstance(rel, str) or not rel.strip():
        msg = (
            f"network edge in {source!r} declares no `relation` — refused "
            f"(da#157: an undeclared relation is never defaulted to STUDIED_UNDER). "
            f"The producer must set `relation` to one of {sorted(VALID_EDGE_RELATIONS)}."
        )
        raise ValueError(msg)
    rel = rel.strip()
    if rel not in VALID_EDGE_RELATIONS:
        msg = (
            f"network edge in {source!r} declares unknown relation {rel!r}; "
            f"expected one of {sorted(VALID_EDGE_RELATIONS)}."
        )
        raise ValueError(msg)
    return rel


_STUDIED_UNDER_QUERY = """\
UNWIND $batch AS row
MATCH (s:Narrator {id: row.from_id})
MATCH (t:Narrator {id: row.to_id})
MERGE (s)-[:STUDIED_UNDER]->(t)
"""

_STUDIED_UNDER_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (s:Narrator {id: row.from_id})
OPTIONAL MATCH (t:Narrator {id: row.to_id})
RETURN row.from_id AS from_id,
       row.to_id AS to_id,
       s IS NOT NULL AS from_exists,
       t IS NOT NULL AS to_exists
"""


def _studied_under_endpoint(name: Any, external_id: Any) -> str | None:
    """Resolve a NETWORK_EDGE endpoint to a canonical Narrator node id.

    Canonical Narrator nodes are keyed ``nar:<uuid5(normalized-name)>``
    (``identity.make_canonical_id``) — by the narrator's *name*, not by the
    source's external id. An edge endpoint must therefore be resolved the same
    way the bio that created the node was: through the name. We key on the
    narrator name (the value that actually matches a node) and only fall back to
    prefixing a pre-canonical id when the name is absent. Resolving by
    ``narrator_node_id(external_id)`` — as an earlier revision did — produced
    ``nar:<source-id>`` ids that match no canonical node, so every edge was
    silently dropped as a missing endpoint.
    """
    if isinstance(name, str) and name.strip():
        return make_canonical_id(normalize_arabic(name))
    if isinstance(external_id, str) and external_id.strip():
        return narrator_node_id(external_id)
    return None


def _load_studied_under(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load STUDIED_UNDER edges from the ``network_edges_*.parquet`` files.

    Globs every ``network_edges_*`` file and keeps only the rows that DECLARE the
    STUDIED_UNDER relation (:func:`_row_relation`, da#133). A NETWORK_EDGE producer
    whose edges are a different relation (e.g. ``mis`` isnad transmission, declared
    TRANSMITTED_TO) is skipped row-by-row regardless of filename. A row that
    declares NO relation (or an unrecognized one) raises ``ValueError`` — it is
    never silently defaulted to STUDIED_UNDER (da#157). Gracefully skips when no
    ``network_edges_*`` file exists.
    """
    edge_files = _parquet_files(staging_dir, "network_edges_")
    if not edge_files:
        logger.info("studied_under_skipped", reason="file_not_found", staging_dir=str(staging_dir))
        return EdgeLoadResult("STUDIED_UNDER", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0

    for path in edge_files:
        for row in _read_parquet_rows(path):
            if _row_relation(row, source=path.name) != EDGE_RELATION_STUDIED_UNDER:
                # A validly-declared non-studentship edge (e.g. an isnad-transmission
                # row → TRANSMITTED_TO) — leave it off STUDIED_UNDER. _row_relation has
                # already REFUSED any row that declares no relation (da#157), so nothing
                # is silently routed here. Not counted as skipped: skipped tracks edges
                # we meant to load but could not resolve, not other relations.
                continue
            from_id = _studied_under_endpoint(
                row.get("from_narrator_name"), row.get("from_external_id")
            )
            to_id = _studied_under_endpoint(row.get("to_narrator_name"), row.get("to_external_id"))
            if not from_id or not to_id:
                skipped += 1
                continue
            # Dedup the same student->teacher pair across source files.
            key = (from_id, to_id)
            if key in seen:
                continue
            seen.add(key)
            batch.append({"from_id": from_id, "to_id": to_id})

    if not batch:
        logger.info("studied_under_no_edges")
        return EdgeLoadResult("STUDIED_UNDER", 0, skipped, 0)

    # Check endpoints
    check_results = _chunked_read(client, _STUDIED_UNDER_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("from_exists") and check.get("to_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_STUDIED_UNDER_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("studied_under_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("STUDIED_UNDER", created, skipped, missing)


# ---------------------------------------------------------------------------
# 6. GRADED_BY — hadith -> grading
# ---------------------------------------------------------------------------

_GRADED_BY_QUERY = """\
UNWIND $batch AS row
MATCH (h:Hadith {id: row.hadith_id})
MATCH (g:Grading {id: row.grading_id})
MERGE (h)-[:GRADED_BY]->(g)
"""

_GRADED_BY_CHECK = """\
UNWIND $batch AS row
OPTIONAL MATCH (h:Hadith {id: row.hadith_id})
OPTIONAL MATCH (g:Grading {id: row.grading_id})
RETURN row.hadith_id AS hadith_id,
       row.grading_id AS grading_id,
       h IS NOT NULL AS hadith_exists,
       g IS NOT NULL AS grading_exists
"""


def _load_graded_by(
    client: Neo4jClient,
    staging_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EdgeLoadResult:
    """Load GRADED_BY edges from hadith staging data.

    Gracefully skips if no hadiths with grades exist.
    """
    files = _parquet_files(staging_dir, "hadiths_")
    if not files:
        logger.info("graded_by_skipped", reason="no_hadith_files")
        return EdgeLoadResult("GRADED_BY", 0, 0, 0)

    batch: list[dict[str, Any]] = []
    skipped = 0
    malformed_ids = 0

    for fp in files:
        rows = _read_parquet_rows(fp)
        for row in rows:
            grade = row.get("grade")
            if not grade:
                continue
            sid = row.get("source_id")
            if not sid:
                skipped += 1
                continue
            try:
                hid = hadith_node_id(sid)
                gid = grading_node_id(sid)
            except DoubledCorpusPrefixError:
                if strict:
                    raise
                skipped += 1
                malformed_ids += 1
                continue
            batch.append({"hadith_id": hid, "grading_id": gid})

    if malformed_ids:
        logger.warning("graded_by_malformed_ids_quarantined", rows=malformed_ids)

    if not batch:
        logger.info("graded_by_no_edges")
        return EdgeLoadResult("GRADED_BY", 0, skipped, 0, malformed_ids=malformed_ids)

    # Check endpoints
    check_results = _chunked_read(client, _GRADED_BY_CHECK, batch, batch_size)
    valid_batch: list[dict[str, Any]] = []
    missing = 0
    for item, check in zip(batch, check_results):
        if check.get("hadith_exists") and check.get("grading_exists"):
            valid_batch.append(item)
        else:
            missing += 1

    created = (
        client.execute_write_batch(_GRADED_BY_QUERY, valid_batch, batch_size=batch_size)
        if valid_batch
        else 0
    )
    logger.info("graded_by_loaded", created=created, skipped=skipped, missing_endpoints=missing)
    return EdgeLoadResult("GRADED_BY", created, skipped, missing, malformed_ids=malformed_ids)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_all_edges(
    client: Neo4jClient,
    staging_dir: Path,
    curated_dir: Path,
    *,
    strict: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EdgeLoadResult]:
    """Load all edge/relationship types into Neo4j.

    Parameters
    ----------
    client:
        Connected Neo4j client.
    staging_dir:
        Directory containing staging Parquet files.
    curated_dir:
        Directory containing curated artifacts — notably
        ``narrator_mentions_resolved.parquet`` (the canonical-id-bearing mentions
        that ``resolve.run_all`` writes here, not to staging). The chain edges
        (NARRATED / TRANSMITTED_TO) read it from here.
    strict:
        If ``True``, raise on missing required files. If ``False``,
        skip gracefully.
    batch_size:
        Number of edges per batch for both read checks and writes.
    """
    results: list[EdgeLoadResult] = []

    # Compute the node loader's real kept set ONCE (da#373) and share it with both
    # chain-edge loaders, so they gate on the same set the node loader loaded —
    # composition AND cross-edition matn dedup — instead of re-deriving a partial
    # gate from the id string. This is what stops a repaired matn key from
    # re-orphaning ~196k dangling chains.
    loaded_hadith_ids = _staging_loaded_hadith_ids(staging_dir)

    results.append(
        _load_transmitted_to(
            client,
            staging_dir,
            curated_dir=curated_dir,
            strict=strict,
            batch_size=batch_size,
            loaded_hadith_ids=loaded_hadith_ids,
        )
    )
    results.append(
        _load_narrated(
            client,
            staging_dir,
            curated_dir=curated_dir,
            strict=strict,
            batch_size=batch_size,
            loaded_hadith_ids=loaded_hadith_ids,
        )
    )
    results.append(_load_appears_in(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_parallel_of(client, staging_dir, strict=strict, batch_size=batch_size))
    results.append(_load_studied_under(client, staging_dir, batch_size=batch_size))
    results.append(_load_graded_by(client, staging_dir, strict=strict, batch_size=batch_size))

    total_created = sum(r.created for r in results)
    total_missing = sum(r.missing_endpoints for r in results)
    logger.info(
        "all_edges_loaded",
        total_created=total_created,
        total_missing_endpoints=total_missing,
        edge_types=len(results),
    )

    # Conformance counter: an edge type that loaded zero edges is a product-level
    # red flag worth surfacing on its own — PARALLEL_OF=0 is exactly the da#160
    # regression (/compare Browse Parallels permanently empty). Emit a WARNING per
    # zero-count type so an empty load is never silent in the run logs.
    zero_types = [r.edge_type for r in results if r.created == 0]
    if zero_types:
        logger.warning(
            "edge_load_conformance",
            zero_count_edge_types=zero_types,
            msg="edge type(s) loaded 0 edges",
        )

    return results
