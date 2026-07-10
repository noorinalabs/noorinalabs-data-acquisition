"""Phase 1: Parsers producing normalized Parquet from raw data.

The set of sources and their run order is the single registry in
:mod:`src.adapters` (epic da#81); :func:`run_all` iterates it, so the parse and
acquire orchestrators can no longer drift on which sources exist.
"""

from __future__ import annotations

from pathlib import Path

from src.adapters import SOURCE_REGISTRY, ParseOutput
from src.parse.identity import DoubledCorpusPrefixError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ParseProducerError(RuntimeError):
    """One or more adapters hit a producer-defect gate during ``parse`` (da#386).

    The da#355 producer gate (``src.parse.base.generate_source_id``) asserts the
    id grammar and raises :exc:`~src.parse.identity.DoubledCorpusPrefixError` when
    ``collection == corpus``. That is a *data-defect* assertion, not a transient
    parse failure: :func:`run_all` used to swallow it in its broad ``except
    Exception`` (it subclasses :exc:`ValueError`), so ``parse`` returned normally
    and exited ``0`` with the **stale** staging parquet on disk — making da#359's
    "re-run ``parse``" remediation a silent no-op.

    :func:`run_all` now collects every such defect and raises this after the parse
    summary so the whole invocation fails loudly; ``_cmd_parse`` maps it to
    :attr:`~src.exit_codes.ExitCode.PARSE_PRODUCER_DEFECT`. Genuinely transient
    per-adapter failures are still logged and skipped (``results[name] = []``) —
    the narrowing is deliberately limited to the producer gate.
    """

    def __init__(self, defects: dict[str, DoubledCorpusPrefixError]) -> None:
        self.defects = dict(defects)
        sources = ", ".join(sorted(self.defects))
        super().__init__(
            f"parse aborted: {len(self.defects)} source(s) hit the da#355 producer "
            f"gate ({sources}). Each source's staging output was NOT (re)written and "
            f"any partial writes were purged. This is a producer data defect — fix "
            f"the upstream source (e.g. the da#353 CSV-stem collection fallback) and "
            f"re-run `parse`; re-running `parse` unchanged will fail here again."
        )


def _normalize_output(result: ParseOutput) -> list[Path]:
    """Normalize parser return values to a flat list of Paths."""
    if isinstance(result, dict):
        return list(result.values())
    if isinstance(result, tuple):
        return list(result)
    if isinstance(result, list):
        return result
    return [result]


def _staging_parquet_state(staging_dir: Path) -> dict[Path, tuple[int, int]]:
    """Snapshot ``(mtime_ns, size)`` of every staging parquet, for partial purge.

    Taken immediately BEFORE an adapter runs so :func:`_purge_partial_writes` can
    identify exactly what that adapter wrote THIS run if it then raises.
    """
    state: dict[Path, tuple[int, int]] = {}
    if not staging_dir.exists():
        return state
    for path in staging_dir.glob("*.parquet"):
        try:
            stat = path.stat()
        except OSError:
            continue
        state[path] = (stat.st_mtime_ns, stat.st_size)
    return state


def _purge_partial_writes(staging_dir: Path, before: dict[Path, tuple[int, int]]) -> list[Path]:
    """Remove staging parquet files created or modified since *before* (da#386).

    When an adapter raises a producer defect it may have already written some of
    its outputs before the gate fired on a later one — a *partial* output set that
    a downstream ``load`` would read as complete. Those files are removed so a
    re-run genuinely re-runs.

    Only files THIS run touched (new, or a changed ``(mtime_ns, size)``) are
    removed. A pre-existing, untouched staging file is not attributable to the
    failing adapter from here, and this code deliberately does NOT re-encode the
    load-side ``{kind}_{slug}.parquet`` glob convention to guess one — a hand-kept
    parallel filename list is the exact fragility the exit-code registry and the
    adapter registry exist to abolish. Such a legacy stale file is overwritten by
    the next *successful* parse of its source (parsers always re-run, never
    skip-if-exists); until the upstream data defect is fixed, the non-zero exit is
    what stops it being read as fresh.
    """
    removed: list[Path] = []
    for path in staging_dir.glob("*.parquet"):
        try:
            now = path.stat()
        except OSError:
            continue
        if before.get(path) != (now.st_mtime_ns, now.st_size):
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def run_all(raw_dir: Path, staging_dir: Path) -> dict[str, list[Path]]:
    """Run all parsers. Continue on transient failure; fail loud on a producer defect.

    Raises :class:`ParseProducerError` (after the summary) if any adapter hit the
    da#355 producer gate, so ``parse`` cannot exit ``0`` with a stale/partial
    staging artifact in place (da#386).
    """
    results: dict[str, list[Path]] = {}
    producer_defects: dict[str, DoubledCorpusPrefixError] = {}
    for adapter in SOURCE_REGISTRY:
        name = adapter.slug
        if not adapter.active:
            # Inactive sources (e.g. open_hadith — a confirmed duplicate, da#191)
            # are never parsed. The registry row is retained for coverage only.
            logger.info("parse_skipped_inactive", source=name)
            continue
        before = _staging_parquet_state(staging_dir)
        try:
            logger.info("parsing", source=name)
            output_files = _normalize_output(adapter.parse(raw_dir, staging_dir))
            results[name] = output_files
            logger.info("parsed", source=name, files=len(output_files))
        except DoubledCorpusPrefixError as exc:
            # da#386: the da#355 producer gate is a data-defect assertion, NOT a
            # transient parse failure. Swallowing it in the broad handler below
            # made `parse` exit 0 with the stale parquet in place, so da#359's
            # "re-run parse" remediation silently changed nothing. Purge whatever
            # this adapter wrote this run (a partial output set), record the
            # defect, and fail the whole `parse` loudly after the summary.
            purged = _purge_partial_writes(staging_dir, before)
            logger.error(
                "parse_producer_defect",
                source=name,
                error=str(exc),
                purged=[str(p) for p in purged],
                exc_info=True,
            )
            results[name] = []
            producer_defects[name] = exc
        except Exception as exc:  # noqa: BLE001
            logger.error("parse_failed", source=name, error=str(exc), exc_info=True)
            results[name] = []

    # Summary with row counts
    import pyarrow.parquet as pq

    print("\n=== Parse Summary ===")
    total_hadiths = 0
    total_mentions = 0
    total_bios = 0
    for name, files in results.items():
        status = "ok" if files else "FAIL"
        row_count = 0
        for f in files:
            if f.exists():
                meta = pq.read_metadata(f)
                row_count += meta.num_rows
                if "hadith" in f.name:
                    total_hadiths += meta.num_rows
                elif "narrator_mention" in f.name:
                    total_mentions += meta.num_rows
                elif "narrator" in f.name and "bio" in f.name:
                    total_bios += meta.num_rows
        print(f"  [{status:4s}] {name:15s}  {len(files):>2d} files  {row_count:>8d} rows")

    print(f"\n  Totals: {total_hadiths} hadiths, {total_mentions} mentions, {total_bios} bios")

    # da#386: fail loud AFTER the summary so the operator sees the full per-source
    # picture, then cannot mistake a producer-gate abort for a clean `parse`.
    if producer_defects:
        raise ParseProducerError(producer_defects)

    return results
