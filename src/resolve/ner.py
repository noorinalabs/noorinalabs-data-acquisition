"""Narrator Named Entity Recognition from isnad chains.

Reads all staging Parquet files and consolidates narrator mentions into a
single ``narrator_mentions_resolved.parquet`` for downstream disambiguation.

Sources with Phase 1 narrator mentions (sanadset, lk) are reused directly.
Arabic-text sources (thaqalayn, open_hadith) use rule-based extraction.
English-only sources (fawaz, sunnah) use keyword-based extraction.
Muhaddithat is skipped (bio/network data only, no raw isnads).
"""

from __future__ import annotations

import csv
import random
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.exit_codes import EXIT_UNROUTED_CORPUS
from src.parse.base import safe_str, write_parquet
from src.parse.name_quality import clean_narrator_name, strip_markup
from src.parse.narrator_extraction import IsnadSegmentationError, extract_narrator_mentions
from src.resolve._inputs import require_input
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA
from src.utils.arabic import is_arabic, normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["EXIT_UNROUTED_CORPUS", "UnroutedCorpusError", "run"]

# Sources that already have Phase 1 narrator_mentions Parquet files.
_PHASE1_MENTION_SOURCES: dict[str, str] = {
    "sanadset": "narrator_mentions_sanadset.parquet",
    "lk": "narrator_mentions_lk.parquet",
}

# Sources with Arabic isnads needing rule-based extraction.
#
# fawaz (da#271): extracted from its Arabic full text, NOT its English translation.
# fawaz ships an empty ``isnad_raw_en``/``isnad_raw_ar`` but a fully-populated
# voweled ``full_text_ar`` (isnad+matn). The old English path produced romanized
# names ("Anas bin Malik") that never share canonical identity with the Arabic
# corpora -- the da#271 cross-script under-merge -- and, because the English
# "Narrated X:" pattern only surfaces the lead companion, it recovered ~1
# narrator/hadith versus the full
# ~5-narrator Arabic chain. Extracting the Arabic text is strictly better than
# transliterating Latin→Arabic (lossy: no short vowels, bin/ibn, apostrophes) or a
# cross-script match-key hack: it yields genuinely Arabic names that merge natively
# AND the complete isnad chain. (sunnah stays English below: its ``full_text_ar``
# is a samāʿ/reading-certificate blob with biographical dates, not a clean isnad.)
#
# NOTE (da#369): the full_text fallback that once mined fawaz's empty-isnad blob
# is removed, so fawaz currently yields 0 mentions; it stays routed here (not an
# unrouted hard-fail) pending the same da#366 isnad/matn split the
# _SPLITTER_DEFERRED_SOURCES corpora below await.
#
# tusi (da#365): a pure routing miss -- ships a fully separated ``isnad_raw_ar``
# (17,089 of 17,421 rows, 98.1%; ~99% carry a transmission phrase, 100% segment
# cleanly), yet was in no route set and so contributed 0 mentions. It needs no
# splitter -- extracted straight from its isnad column like any Arabic corpus. Its
# ~332 null-isnad rows hit the da#369 null-skip, never the removed matn mine.
_ARABIC_SOURCES: set[str] = {"thaqalayn", "open_hadith", "fawaz", "tusi"}

# Sources with English text needing keyword-based extraction.
_ENGLISH_SOURCES: set[str] = {"sunnah"}

# Sources whose isnad chains are embedded INSIDE ``full_text_ar`` with no separate
# ``isnad_raw_ar`` column (halimbahae and bihar are both 100% null-isnad yet
# isnad-bearing in full_text). They CANNOT be extracted until a validated
# isnad/matn splitter (da#366) lands: routing them through NER today extracts
# nothing (null isnad column), and reinstating the removed full_text fallback would
# re-import the matn pollution da#369 deleted (main#352). They are EXPLICITLY routed
# to this deferred bucket -- not left unrouted -- so the da#369 hard-fail does not
# fire on a known-held corpus, while a *genuinely* unrouted corpus (in no bucket at
# all) still fails loud. They contribute 0 mentions until da#366: better silent-but-
# declared than matn-polluted (#928). The value is the issue blocking each corpus.
_SPLITTER_DEFERRED_SOURCES: dict[str, str] = {
    "halimbahae": "da#366",
    "bihar": "da#366",
}

# Sources to skip entirely -- no isnad chain is reachable via NER over hadith rows.
#   muhaddithat: bio/network data only, no raw isnads.
#   mis (da#365): its ``hadiths_mis.parquet`` rows carry a null ``isnad_raw_ar`` and
#     no ``full_text_ar``; mis's transmission chains are staged separately as
#     ``network_edges_mis.parquet`` and read by the STUDIED_UNDER glob but skipped
#     (they declare TRANSMITTED_TO), so no mis edge is persisted — produced, not
#     loaded as a graph edge — pending a name-reconciliation crosswalk (da#364),
#     never mined from raw isnad text here. Routed to skip (not unrouted) so the
#     da#369 hard-fail does not fire; it contributes 0 NER mentions by construction.
_SKIP_SOURCES: set[str] = {"muhaddithat", "mis"}

# The union of every corpus with an explicit NER route. A staged corpus outside
# this set has no way to have its narrators extracted and MUST fail loud (da#369)
# rather than be silently dropped or (pre-da#369) matn-mined via the removed
# full_text fallback. da#365 completes the routing: every corpus that stages a
# globbed NER input (hadiths_* / narrator_mentions_*) now declares a route --
# extract (phase1 / arabic / english), splitter-deferred, or skip. A brand-new
# staged corpus in none of these buckets still trips the guard.
_ROUTED_CORPORA: frozenset[str] = frozenset(
    set(_PHASE1_MENTION_SOURCES)
    | _ARABIC_SOURCES
    | _ENGLISH_SOURCES
    | set(_SPLITTER_DEFERRED_SOURCES)
    | _SKIP_SOURCES
)


class UnroutedCorpusError(BaseException):
    """A corpus is staged for resolution but has no NER extraction route (da#369).

    Subclasses ``BaseException`` (not ``Exception``) **on purpose**, exactly like
    :class:`~src.resolve._deps.MissingDependencyError` and
    :class:`~src.resolve._checkpoint.StopAfterReached`: :func:`src.resolve.run_all`
    wraps every stage in ``except Exception`` and logs-and-continues, so an
    ``Exception`` raised here would be swallowed as a failed NER step and the
    pipeline would march on and report success on a resolve that silently omitted
    the corpus. Raised as a ``BaseException`` so it sails through those guards and
    surfaces to the CLI, which maps it to :data:`EXIT_UNROUTED_CORPUS`.

    Before da#369, an unrouted (or isnad-null) corpus did not raise: the extractor
    fell back to mining ``full_text_ar`` (isnad+matn), producing 380,771
    matn-derived pseudo-narrator mentions (11.7% of all mentions) and 38,262
    narrators that existed ONLY in matn text. That fallback is removed; NER
    extraction is per-corpus opt-in and an unrouted staged corpus fails loud so a
    proper route is added (da#365) rather than silently matn-mined.
    """

    def __init__(self, unrouted: set[str], routed: frozenset[str]) -> None:
        self.unrouted = sorted(unrouted)
        self.routed = sorted(routed)
        super().__init__(
            f"staged corpora with no NER extraction route: {self.unrouted}. "
            f"Routed corpora: {self.routed}. Every staged corpus must be explicitly "
            "opted in to a NER route (phase1 / arabic / english), the "
            "splitter-deferred bucket (da#366), or the skip list — add a "
            "per-corpus route (da#365). NER never falls back to mining "
            "full_text_ar (matn) for an unrouted or isnad-null corpus (da#369)."
        )


def _discover_staged_corpora(staging_dir: Path) -> set[str]:
    """Enumerate the corpora actually present in staging.

    Reads each staged Parquet's ``source_corpus`` column rather than parsing the
    filename, so a mis-named file cannot hide an unrouted corpus. NER's own
    resolved output is excluded, and a file without a ``source_corpus`` column is
    skipped (it carries no corpus claim to route).
    """
    corpora: set[str] = set()
    for pattern in ("hadiths_*.parquet", "narrator_mentions_*.parquet"):
        for path in sorted(staging_dir.glob(pattern)):
            if path.name == "narrator_mentions_resolved.parquet":
                continue  # NER's own output, not a staged input
            if "source_corpus" not in pq.read_schema(path).names:
                continue
            table = pq.read_table(path, columns=["source_corpus"])
            for value in table.column("source_corpus").to_pylist():
                if value:
                    corpora.add(str(value))
    return corpora


def _load_phase1_mentions(
    staging_dir: Path,
    corpus: str,
    filename: str,
    *,
    required: bool = False,
) -> list[dict[str, str | int | None]]:
    """Load pre-extracted Phase 1 narrator mentions and map to resolved schema.

    ``required`` (da#361): when the corpus is actually staged (present in
    ``_discover_staged_corpora``) its Phase-1 mentions file MUST exist — its
    absence is an upstream defect (parse mis-wrote or never ran), not an empty
    result, so :func:`require_input` raises. When the corpus is merely a
    *configured* Phase-1 source that is not part of this (partial) staging subset,
    ``required`` is ``False`` and the absent file is a legitimate skip: ``run``
    loops over every configured source, and resolving a subset of corpora is a
    supported mode (this is why ``run`` on an empty staging dir must NOT raise).
    """
    path = staging_dir / filename
    if not path.exists():
        require_input(
            stage="ner",
            present=not required,
            input_desc=f"{filename} (Phase-1 mentions for staged corpus {corpus!r})",
            produced_by="parse (Phase-1 mention extractor)",
            remediation=(
                f"corpus {corpus!r} is staged but its Phase-1 mentions file is absent; "
                "re-run `parse` for it"
            ),
        )
        # required is False: a configured Phase-1 source simply absent from this
        # (partial) staging subset — a legitimate skip, not a defect.
        logger.info("phase1_mentions_absent_unstaged", corpus=corpus, path=str(path))
        return []

    table = pq.read_table(path)
    rows: list[dict[str, str | int | None]] = []
    dropped = 0

    # Chain identity (da#282): sanadset/lk emit ``chain_index`` on their mention
    # rows; carry it through so the loader groups per-chain, not per-hadith. Read
    # defensively so a legacy pre-da#282 parse file (no such column) still loads —
    # a missing column reads back as chain 0 (single chain).
    has_chain_index = "chain_index" in table.column_names

    for i in range(table.num_rows):
        name_ar = safe_str(table.column("name_ar")[i].as_py())
        name_en = safe_str(table.column("name_en")[i].as_py())
        name_ar_norm = safe_str(table.column("name_ar_normalized")[i].as_py())

        # Use Arabic name if available, else English.
        name_raw = name_ar or name_en
        name_normalized = name_ar_norm or (normalize_arabic(name_ar) if name_ar else name_en)

        # Name-quality filter (da#247): strip markup / honorifics and drop
        # non-name spans (mubham descriptors, mis-parsed text). sanadset's coarse
        # <NAR> firehose is the main pollution source here.
        cleaned = clean_narrator_name(name_normalized)
        if cleaned is None:
            dropped += 1
            continue
        name_normalized = cleaned
        name_raw = strip_markup(name_raw) or name_raw

        chain_index = table.column("chain_index")[i].as_py() if has_chain_index else 0
        rows.append(
            {
                "mention_id": str(uuid.uuid4()),
                "hadith_id": table.column("source_hadith_id")[i].as_py(),
                "source_corpus": corpus,
                "position_in_chain": table.column("position_in_chain")[i].as_py(),
                "chain_index": chain_index if chain_index is not None else 0,
                "name_raw": name_raw,
                "name_normalized": name_normalized,
                "canonical_narrator_id": None,
                "transmission_method": safe_str(table.column("transmission_method")[i].as_py()),
                "confidence": None,
            }
        )

    logger.info("phase1_mentions_loaded", corpus=corpus, mentions=len(rows), dropped=dropped)
    return rows


def _extract_from_hadiths(
    staging_dir: Path,
    corpus: str,
    language: str,
    *,
    required: bool = False,
) -> list[dict[str, str | int | None]]:
    """Extract narrator mentions from hadith Parquet files for a given corpus.

    ``required`` (da#361): same discrimination as :func:`_load_phase1_mentions`. A
    staged corpus (present in ``_discover_staged_corpora``) routed here MUST have
    its ``hadiths_{corpus}*.parquet`` — its absence is an upstream defect and
    :func:`require_input` raises. A configured Arabic/English source that is not in
    this staging subset (``required=False``) is a legitimate skip.
    """
    pattern = f"hadiths_{corpus}*.parquet"
    hadith_files = sorted(staging_dir.glob(pattern))
    if not hadith_files:
        require_input(
            stage="ner",
            present=not required,
            input_desc=f"{pattern} (isnad text for staged corpus {corpus!r})",
            produced_by="parse (hadith adapters)",
            remediation=(
                f"corpus {corpus!r} is staged but has no {pattern}; re-run `parse` for it"
            ),
        )
        # required is False: configured source absent from this staging subset.
        logger.info("hadiths_absent_unstaged", corpus=corpus, pattern=pattern)
        return []

    rows: list[dict[str, str | int | None]] = []
    null_isnad_count = 0
    unsegmentable_count = 0
    dropped_names = 0
    total_hadiths = 0

    for hf in hadith_files:
        table = pq.read_table(hf)
        total_hadiths += table.num_rows

        isnad_col = "isnad_raw_ar" if language == "ar" else "isnad_raw_en"

        for i in range(table.num_rows):
            hadith_id = table.column("source_id")[i].as_py()
            isnad_text = safe_str(table.column(isnad_col)[i].as_py())

            # da#369: NER extracts ONLY from the dedicated isnad column. It no
            # longer falls back to mining ``full_text_ar`` (isnad+matn) when the
            # isnad is null — that path produced 380,771 matn-derived pseudo-
            # narrator mentions (11.7% of all mentions) and 38,262 narrators that
            # existed ONLY in matn text. A null isnad is skip-and-counted here; a
            # corpus whose isnad column is genuinely empty must be given a proper
            # per-corpus route (da#365), never silently matn-mined.
            if not isnad_text:
                null_isnad_count += 1
                continue

            # The segmenter fails LOUD on an isnad it cannot split into narrators
            # (da#158) — e.g. a row whose isnad field is actually matn, or an
            # isnad+matn blob (da#244: thaqalayn hadith 5762). That is a per-row
            # data defect, not a pipeline-fatal one: skip-and-count this hadith so
            # one bad row does not abort the entire NER pass (which would cascade
            # to skip disambiguate and strand the whole resolve). The fail-loud
            # guard is by design; its consumer must tolerate per-row rejection.
            try:
                spans = extract_narrator_mentions(isnad_text, language)
            except IsnadSegmentationError:
                unsegmentable_count += 1
                logger.warning(
                    "hadith_isnad_unsegmentable",
                    corpus=corpus,
                    hadith_id=hadith_id,
                    isnad_preview=isnad_text[:80],
                )
                continue
            for span in spans:
                name_raw = span.name
                if language == "ar":
                    name_normalized = normalize_arabic(name_raw)
                else:
                    name_normalized = name_raw.strip()

                # Name-quality filter (da#247): the token-count cap here is the
                # backstop for the thaqalayn parser dumping whole hadith bodies
                # into the name field; markup / mubham guards apply too.
                cleaned = clean_narrator_name(name_normalized)
                if cleaned is None:
                    dropped_names += 1
                    continue
                name_normalized = cleaned
                name_raw = strip_markup(name_raw) or name_raw

                rows.append(
                    {
                        "mention_id": str(uuid.uuid4()),
                        "hadith_id": hadith_id,
                        "source_corpus": corpus,
                        "position_in_chain": span.position,
                        # One isnad column per hadith → a single linear chain
                        # (the segmenter does not split parallel sub-chains), so
                        # chain 0 (da#282).
                        "chain_index": 0,
                        "name_raw": name_raw,
                        "name_normalized": name_normalized,
                        "canonical_narrator_id": None,
                        "transmission_method": span.transmission_method,
                        "confidence": None,
                    }
                )

    null_pct = (null_isnad_count / total_hadiths * 100) if total_hadiths else 0.0
    logger.info(
        "extraction_complete",
        corpus=corpus,
        language=language,
        total_hadiths=total_hadiths,
        null_isnad_pct=round(null_pct, 1),
        unsegmentable_skipped=unsegmentable_count,
        dropped_names=dropped_names,
        mentions_extracted=len(rows),
        mentions_per_hadith=round(len(rows) / max(total_hadiths, 1), 2),
    )
    return rows


def _write_name_audit_csv(
    rows: list[dict[str, str | int | None]],
    output_dir: Path,
    sample_size: int = 100,
) -> Path:
    """Export a random sample of name_raw vs name_normalized for manual audit."""
    audit_path = output_dir / "ner_name_audit.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows that have both raw and normalized names.
    candidates = [r for r in rows if r.get("name_raw") and r.get("name_normalized")]
    rng = random.Random(42)
    sample = rng.sample(candidates, min(sample_size, len(candidates)))

    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_corpus", "name_raw", "name_normalized", "is_arabic"])
        for r in sample:
            raw = str(r["name_raw"])
            writer.writerow([r["source_corpus"], raw, r["name_normalized"], is_arabic(raw)])

    logger.info("name_audit_written", path=str(audit_path), rows=len(sample))
    return audit_path


def run(staging_dir: Path, output_dir: Path) -> list[Path]:
    """Extract narrator mentions from parsed isnad chains.

    Reads staging Parquet files and produces resolved narrator mention tables.
    Returns list of output file paths.

    Crash-resume (da#272): NER is intentionally NOT intra-stage-checkpointed. Each
    run re-mints ``mention_id`` via ``uuid.uuid4()``, so a partial resume would
    stitch two id-spaces into one mentions file and corrupt the mention-keyed
    disambiguate outputs. NER is also the cheap stage (~2 min). Its "resume" is
    therefore *reusing the whole existing mentions file*, which the CLI expresses
    as ``resolve --from-step disambiguate``: NER is skipped entirely and the
    already-written ``narrator_mentions_resolved.parquet`` (and disambiguate's
    checkpoint keyed off it) is reused verbatim. See ``run_all(from_step=...)``.
    """
    logger.info("ner_run_start", staging_dir=str(staging_dir), output_dir=str(output_dir))

    # da#369: hard-fail on a staged corpus that has no NER extraction route,
    # BEFORE any extraction or output is written. This replaces the removed
    # full_text (matn) fallback: rather than silently mining ``full_text_ar`` for
    # an unrouted or isnad-null corpus, resolution aborts loudly so the corpus is
    # given a proper per-corpus route (da#365). Raised as a ``BaseException`` so it
    # sails through ``run_all``'s per-stage ``except Exception`` to the CLI.
    staged = _discover_staged_corpora(staging_dir)
    unrouted = staged - _ROUTED_CORPORA
    if unrouted:
        raise UnroutedCorpusError(unrouted, _ROUTED_CORPORA)

    all_rows: list[dict[str, str | int | None]] = []

    # da#361: a configured source is *required* only when it is actually staged.
    # ``run`` loops over every configured source, but resolving a partial staging
    # subset is supported, so an absent file for a non-staged source is a legitimate
    # skip while an absent file for a STAGED corpus is an upstream defect that must
    # fail loud (``require_input`` inside each helper). ``staged`` is already
    # computed above for the da#369 routing guard.

    # Step 1: Load Phase 1 pre-extracted mentions (sanadset, lk).
    for corpus, filename in _PHASE1_MENTION_SOURCES.items():
        rows = _load_phase1_mentions(staging_dir, corpus, filename, required=corpus in staged)
        all_rows.extend(rows)

    # Step 2: Extract from Arabic-text sources.
    for corpus in sorted(_ARABIC_SOURCES):
        rows = _extract_from_hadiths(staging_dir, corpus, language="ar", required=corpus in staged)
        all_rows.extend(rows)

    # Step 3: Extract from English-only sources.
    for corpus in sorted(_ENGLISH_SOURCES):
        rows = _extract_from_hadiths(staging_dir, corpus, language="en", required=corpus in staged)
        all_rows.extend(rows)

    # Step 4: Log deferred sources -- isnad-in-matn corpora explicitly held until
    # the da#366 splitter lands. Routed (no hard-fail) but not extracted; logged
    # loudly so their absence from the graph is visible, never a silent omission.
    for corpus, blocked_on in sorted(_SPLITTER_DEFERRED_SOURCES.items()):
        logger.info(
            "ner_deferred_source",
            corpus=corpus,
            blocked_on=blocked_on,
            reason="isnad_in_matn_needs_splitter",
        )

    # Step 5: Log skipped sources.
    for corpus in sorted(_SKIP_SOURCES):
        logger.info("ner_skip_source", corpus=corpus, reason="no_raw_isnads")

    # Step 6: Per-source metrics summary.
    source_counts: dict[str, int] = {}
    for r in all_rows:
        src = str(r["source_corpus"])
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        logger.info("ner_source_summary", source_corpus=src, total_mentions=count)
    logger.info("ner_total_mentions", total=len(all_rows))

    # Step 7: Build output table.
    output_paths: list[Path] = []

    if all_rows:
        arrays: dict[str, pa.Array] = {
            "mention_id": pa.array([r["mention_id"] for r in all_rows], type=pa.string()),
            "hadith_id": pa.array([r["hadith_id"] for r in all_rows], type=pa.string()),
            "source_corpus": pa.array([r["source_corpus"] for r in all_rows], type=pa.string()),
            "position_in_chain": pa.array(
                [r["position_in_chain"] for r in all_rows], type=pa.int32()
            ),
            "chain_index": pa.array([r.get("chain_index", 0) for r in all_rows], type=pa.int32()),
            "name_raw": pa.array([r["name_raw"] for r in all_rows], type=pa.string()),
            "name_normalized": pa.array([r["name_normalized"] for r in all_rows], type=pa.string()),
            "canonical_narrator_id": pa.array(
                [r["canonical_narrator_id"] for r in all_rows], type=pa.string()
            ),
            "transmission_method": pa.array(
                [r["transmission_method"] for r in all_rows], type=pa.string()
            ),
            "confidence": pa.array([r["confidence"] for r in all_rows], type=pa.float32()),
        }
        table = pa.table(arrays, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA)
        resolved_path = output_dir / "narrator_mentions_resolved.parquet"
        write_parquet(table, resolved_path, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA)
        output_paths.append(resolved_path)
    else:
        logger.warning("ner_no_mentions", msg="No narrator mentions extracted from any source")

    # Step 8: Name audit CSV.
    if all_rows:
        audit_path = _write_name_audit_csv(all_rows, output_dir)
        output_paths.append(audit_path)

    logger.info("ner_run_complete", output_files=[str(p) for p in output_paths])
    return output_paths
