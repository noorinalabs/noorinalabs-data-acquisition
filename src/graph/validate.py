"""Graph validation query runner."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger
from src.utils.neo4j_client import Neo4jClient

logger = get_logger(__name__)

__all__ = ["register_classifier", "run_validation", "ValidationResult"]

_DEFAULT_DEVIATION_THRESHOLD = 10.0

# Post-load validation must be *bounded*: on the full graph (~576k chains,
# ~2.5M TRANSMITTED_TO edges) the chain_integrity variable-length cycle
# traversal degrades super-linearly and never returns (>40 min), which is why
# stg+prod loads (main#723) were forced to run with --skip-validation. A slow or
# cyclic graph must therefore *downgrade* validation to a non-fatal warning
# rather than hang the loader — the load itself already succeeded; validation is
# a post-load check. Default: 5 minutes per query. da#259.
_DEFAULT_VALIDATION_TIMEOUT_SECONDS = 300.0


class _ValidationTimeout(Exception):
    """Raised internally when a single validation query exceeds its time budget.

    Carries both the wall-clock backstop and the server-side statement-timeout
    cases so the runner can downgrade either to a warning uniformly.
    """


def _is_timeout_error(exc: BaseException) -> bool:
    """True if *exc* is a Neo4j server-side transaction/statement timeout.

    Neo4j reports a tripped transaction timeout with a status code ending in
    ``TransactionTimedOut``; treat any timeout-flavoured code as a downgrade.
    """
    code = str(getattr(exc, "code", "") or "")
    return "TimedOut" in code or "Timeout" in code


def _execute_read_bounded(
    client: Neo4jClient,
    cypher_text: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Run a read query with a hard upper bound on wall-clock time.

    Defence in depth, because "never hang the loader" must hold even if one
    layer misbehaves:

    1. **Server-side statement timeout** — ``timeout_seconds`` is handed to the
       driver so Neo4j itself aborts the transaction and releases resources.
    2. **Client-side wall-clock backstop** — the query runs on a *daemon* thread
       joined for ``timeout_seconds``; if it is still alive we abandon it (the
       daemon dies with the process, and the server-side timeout reclaims the
       query) and raise :class:`_ValidationTimeout`.

    A server-side timeout surfaces as a Neo4j error inside the worker thread and
    is re-raised here as :class:`_ValidationTimeout`; any other error propagates
    unchanged for the caller to classify as a hard failure.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["rows"] = client.execute_read(cypher_text, timeout=timeout_seconds)
        except BaseException as exc:  # noqa: BLE001 - marshalled back to caller thread
            box["error"] = exc

    thread = threading.Thread(target=_run, name="validation-query", daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise _ValidationTimeout(f"wall-clock timeout after {timeout_seconds:.0f}s")
    if "error" in box:
        err = box["error"]
        if _is_timeout_error(err):
            raise _ValidationTimeout(str(err)) from err
        raise err
    rows: list[dict[str, Any]] = box.get("rows", [])
    return rows


def _mask_line_comments(text: str) -> str:
    """Blank out ``//``-to-end-of-line comment content, character-for-character.

    Returns a string of identical length (and identical newline positions) to
    *text*, with every character inside a ``//`` comment replaced by a space.
    Used only to locate statement-separating ``;`` characters — indices found
    in the masked string slice correctly into the original.

    This exists because the validation ``.cypher`` files' prose comments
    themselves contain semicolons (e.g. ``// ...decision must cover; it is
    contractual, not a defect.`` in ``graph_integrity_deferred_inventory.cypher``)
    — a naive ``text.split(";")`` mis-splits a single comment sentence into two
    "statements" and Neo4j rejects the resulting fragment as invalid Cypher.
    Masking comments first means only a ``;`` in actual code counts.
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def _split_statements(cypher_text: str) -> list[str]:
    """Split a ``.cypher`` file's text into individual top-level statements.

    The Neo4j driver enforces "exactly one statement per query" (da#319): a
    ``.cypher`` file that packs several semicolon-separated blocks — e.g. the
    inventory reports ``graph_integrity_deferred_inventory.cypher`` (ADR-004)
    and ``sanadset_orphan_inventory.cypher`` (ADR-003), each a 5-statement
    sizing script — raises ``CypherSyntaxError: Expected exactly one statement
    per query but got: N`` when read as one blob and executed in a single
    ``tx.run()``.

    These validation files never embed a semicolon inside a string literal, so
    splitting on ``;`` is safe *for code* — but their ``//`` comments do contain
    prose semicolons, so the split must ignore comment content (see
    :func:`_mask_line_comments`; this was caught by the real-Neo4j integration
    test, not the canned-row unit stubs). This is intentionally generic: ANY
    ``.cypher`` file under ``queries/validation/`` with more than one statement
    is handled the same way, not just the two named above. A single-statement
    file with no code-level ``;`` splits into exactly one entry, identical to
    the pre-da#319 behavior.
    """
    masked = _mask_line_comments(cypher_text)
    statements: list[str] = []
    start = 0
    for idx, ch in enumerate(masked):
        if ch != ";":
            continue
        _append_statement(statements, cypher_text[start:idx])
        start = idx + 1
    _append_statement(statements, cypher_text[start:])
    return statements


def _append_statement(statements: list[str], candidate: str) -> None:
    """Append *candidate* to *statements* unless it is blank or comment-only."""
    stmt = candidate.strip()
    if not stmt:
        return
    # Drop comment-only segments (e.g. a file-level header block that happens
    # to end at a `;` boundary) -- not an executable statement.
    has_code = any(line.strip() and not line.strip().startswith("//") for line in stmt.splitlines())
    if has_code:
        statements.append(stmt)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation query."""

    query_name: str
    passed: bool
    details: str
    row_count: int
    warning: bool = False

    @property
    def is_fatal(self) -> bool:
        """A hard failure — not a pass and not a downgraded/timed-out warning."""
        return not self.passed and not self.warning

    @property
    def status(self) -> str:
        """Display status: ``PASS`` / ``FAIL`` / ``WARN``."""
        if self.warning:
            return "WARN"
        return "PASS" if self.passed else "FAIL"


ClassifierFunc = Callable[
    [str, list[dict[str, object]], float],
    ValidationResult,
]


def _classify_orphan_narrators(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    count = len(rows)
    passed = count == 0
    details = "no orphans" if passed else f"{count} orphan narrator(s) found"
    return ValidationResult(query_name, passed, details, count)


_CHAIN_INTEGRITY_COVERAGE = (
    "coverage: cycle length 1 (self-loop) and length 2 (reciprocal pair) counted "
    "exactly; cycles of length >= 3 are NOT measured"
)


def _classify_chain_integrity(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Classify the chain-integrity census (da#250).

    ``chain_integrity.cypher`` returns exactly one aggregate row. Two distinct
    signals come out of it:

    * ``self_loops`` is the **gate**. A narrator who transmits to himself is
      unambiguously corrupt. 0 on stg and prod today, so this guard is live and
      non-vacuous rather than decorative.
    * ``reciprocal_pairs`` is a **metric**, never a gate. Reciprocal
      transmission (A taught B *and* B taught A) is historically impossible, so
      every pair is manufactured upstream by identity collapse in ``resolve``
      (da#356) — it is a thermometer for over-merge, not a defect the loader can
      or should fix. Failing the load on it would only punish the loader for
      someone else's merge. Re-measure after da#356 lands (da#248).

    The predecessor counted ``len(rows)`` against a query ending in ``LIMIT
    100``, so on the full graph it reported the constant ``100 cycle(s)
    detected`` — a cap presented as a measurement, and the rc=1 that main#723
    misread as a data defect. It also passed on ``[]``: an aggregate query that
    returns no row did not run, and a zero you never measured is not a zero.
    """
    if len(rows) != 1:
        return ValidationResult(
            query_name,
            passed=False,
            details=(
                f"census did not run: expected exactly 1 aggregate row, got {len(rows)}. "
                "An absent row is not a zero."
            ),
            row_count=0,
        )

    row = rows[0]
    try:
        self_loops = int(row["self_loops"])  # type: ignore[call-overload]
        reciprocal_pairs = int(row["reciprocal_pairs"])  # type: ignore[call-overload]
    except (KeyError, TypeError, ValueError):
        return ValidationResult(
            query_name,
            passed=False,
            details=(
                f"malformed census row, columns={sorted(row)} — "
                "expected self_loops, reciprocal_pairs"
            ),
            row_count=0,
        )

    passed = self_loops == 0
    gate = (
        "0 self-loops"
        if passed
        else f"{self_loops} self-loop narrator(s) — a narrator transmitting to himself"
    )
    details = (
        f"{gate}; {reciprocal_pairs} reciprocal pair(s) "
        "[metric, not a gate: upstream over-merge, da#248/da#356]; "
        f"{_CHAIN_INTEGRITY_COVERAGE}"
    )
    return ValidationResult(query_name, passed, details, self_loops + reciprocal_pairs)


def _classify_transmitted_to_hadith_ref(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Regression guard for da#325: every edge ``hadith_id`` must resolve.

    The query returns the distinct ``TRANSMITTED_TO.hadith_id`` values that match
    no ``Hadith`` node — the exact failure that made ``GET /graph/hadith/{id}/
    chain`` empty for every hadith (raw ``sanadset:...`` edge ids vs canonical
    ``hdt:sanadset:...`` node ids). 0 rows = every edge id references a Hadith.
    """
    count = len(rows)
    passed = count == 0
    details = (
        "all TRANSMITTED_TO.hadith_id reference a Hadith node"
        if passed
        else f"{count} TRANSMITTED_TO.hadith_id value(s) match no Hadith node (da#325)"
    )
    return ValidationResult(query_name, passed, details, count)


def _classify_collection_coverage(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Compare expected vs actual hadith counts per collection.

    A collection fails the gate two ways:

    * ``deviation_pct`` exceeds ``deviation_threshold`` — the loaded count
      strayed too far from the collection's ``expected_count``.
    * ``deviation_pct`` is NULL — the collection carries no ``expected_count``
      (the cypher's ``CASE ... ELSE null`` arm), so coverage is **un-evaluable**.
      The predecessor *skipped* those rows and then reported "all within
      threshold", so a collection that loaded 0 of its 650,986 rows was rolled
      into a green summary — a false PASS hiding a total load failure (da#382).
      An un-evaluable collection is not a healthy one: it fails the gate. Only a
      numeric deviation within threshold is a positive PASS.
    """
    count = len(rows)
    failures: list[str] = []
    for row in rows:
        cid = row.get("collection_id", "?")
        dev = row.get("deviation_pct")
        if isinstance(dev, int | float):
            if dev > deviation_threshold:
                failures.append(f"{cid}: {dev:.1f}% deviation")
        else:
            actual = row.get("actual", "?")
            failures.append(
                f"{cid}: expected_count missing — coverage un-evaluable (actual={actual})"
            )
    passed = len(failures) == 0
    details = "all within threshold" if passed else "; ".join(failures)
    return ValidationResult(query_name, passed, details, count)


def _row_count_of(row: dict[str, object]) -> int:
    """Extract the ``row_count`` int a block-summary dict carries (0 if absent/odd-shaped)."""
    value = row.get("row_count", 0)
    return value if isinstance(value, int) else 0


def _is_block_summary_rows(rows: list[dict[str, object]]) -> bool:
    """True if *rows* is the per-block summary shape (not a flat real row-set).

    For a multi-statement file, ``run_validation`` cannot present a single flat
    row-set to a classifier, so it hands over one summary dict per statement,
    constructed with **exactly** the keys ``{"block", "row_count"}``. An exact
    key-set match distinguishes those from real Cypher result rows (which carry
    the query's actual column names), so the single-statement path — where the
    real rows are passed straight through — is never mis-detected. Empty input
    is not a block-summary (there are no blocks to sum).
    """
    return bool(rows) and all(set(row.keys()) == {"block", "row_count"} for row in rows)


def _classify_default(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Default classifier — pass if 0 rows (conservative).

    For a multi-statement file with no dedicated classifier, ``run_validation``
    hands over one per-block summary dict per statement rather than a flat
    row-set. In that shape the true result-row count is the **sum** of each
    block's ``row_count``, not ``len(rows)`` — the latter is the *block* count,
    which for a ≥2-statement file is never 0 and would make the file
    unconditionally FAIL, relocating the exact da#319 "clean load trips
    sys.exit(1)" bug to the next multi-statement validation file added. When the
    input is the block-summary shape, apply zero-is-pass to the summed total;
    otherwise keep the single-statement ``len(rows)`` semantics byte-for-byte.
    """
    if _is_block_summary_rows(rows):
        count = sum(_row_count_of(row) for row in rows)
    else:
        count = len(rows)
    passed = count == 0
    details = f"{count} row(s) returned" if count else "empty result"
    return ValidationResult(query_name, passed, details, count)


def _classify_informational_inventory(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Always-PASS classifier for owner-accepted inventory/sizing reports.

    ``graph_integrity_deferred_inventory`` (ADR-004) and
    ``sanadset_orphan_inventory`` (ADR-003) are READ-ONLY, multi-statement
    ``.cypher`` files whose whole purpose is to SIZE a tracked, owner-accepted
    nonzero count — they are not remediated by, nor a defect surfaced by,
    running the file (da#319). ``run_validation`` feeds this classifier one
    summary dict per top-level statement (``block`` 1-indexed, ``row_count``),
    since a multi-statement file has no single flat row-set to apply the
    0-rows-is-pass default to. This mirrors the ``hadith_number_coverage``
    precedent in ``src/parse/validate.py`` — an unconditional PASS that
    reports a metric instead of gating on it.
    """
    total = sum(_row_count_of(row) for row in rows)
    if not rows:
        details = "no statement blocks executed (informational, always PASS)"
    else:
        breakdown = "; ".join(
            f"block {row.get('block')}: {row.get('row_count')} row(s)" for row in rows
        )
        details = (
            f"{total} row(s) across {len(rows)} block(s) (informational, always PASS) — {breakdown}"
        )
    return ValidationResult(query_name, passed=True, details=details, row_count=total)


_CLASSIFIER_REGISTRY: dict[str, ClassifierFunc] = {
    "orphan_narrators": _classify_orphan_narrators,
    "chain_integrity": _classify_chain_integrity,
    "transmitted_to_hadith_ref": _classify_transmitted_to_hadith_ref,
    "collection_coverage": _classify_collection_coverage,
    "graph_integrity_deferred_inventory": _classify_informational_inventory,
    "sanadset_orphan_inventory": _classify_informational_inventory,
}


def register_classifier(name: str, func: ClassifierFunc) -> None:
    """Register a custom classifier for a query name.

    This allows downstream code to add new validation classifiers without
    modifying this module directly.
    """
    _CLASSIFIER_REGISTRY[name] = func


def _classify(
    query_name: str,
    rows: list[dict[str, object]],
    *,
    deviation_threshold: float = _DEFAULT_DEVIATION_THRESHOLD,
) -> ValidationResult:
    """Classify query results as pass/fail based on query semantics."""
    classifier = _CLASSIFIER_REGISTRY.get(query_name, _classify_default)
    return classifier(query_name, rows, deviation_threshold)


def run_validation(
    client: Neo4jClient,
    queries_dir: Path,
    *,
    deviation_threshold: float = _DEFAULT_DEVIATION_THRESHOLD,
    timeout_seconds: float = _DEFAULT_VALIDATION_TIMEOUT_SECONDS,
) -> list[ValidationResult]:
    """Run all ``.cypher`` files in ``queries_dir/validation/``.

    A file may contain multiple ``;``-separated top-level statements (da#319)
    — the Neo4j driver rejects more than one statement per ``tx.run()`` call,
    so each file is split via :func:`_split_statements` and every statement is
    executed as its own bounded query, then aggregated back into a single
    :class:`ValidationResult` per file.

    Each statement is executed with a per-statement time budget
    (``timeout_seconds``). A statement that exceeds it — e.g. the
    ``chain_integrity`` cycle traversal on a slow or cyclic full graph — is
    **downgraded to a non-fatal warning** instead of hanging the loader
    (da#259). The load itself has already succeeded at this point; validation
    is a post-load check, so a timed-out check is inconclusive, not a failure.

    Parameters
    ----------
    client:
        Connected Neo4j client.
    queries_dir:
        Root queries directory (must contain a ``validation/`` subdirectory).
    deviation_threshold:
        Maximum acceptable deviation percentage for collection coverage.
    timeout_seconds:
        Per-query wall-clock/statement-timeout budget. On overrun the query is
        recorded as a ``WARN`` (``warning=True``), which is non-fatal.

    Returns
    -------
    list[ValidationResult]
        One result per query file executed.
    """
    validation_dir = queries_dir / "validation"
    if not validation_dir.is_dir():
        logger.warning("validation_dir_missing", path=str(validation_dir))
        return []

    cypher_files = sorted(validation_dir.glob("*.cypher"))
    if not cypher_files:
        logger.warning("no_validation_queries", path=str(validation_dir))
        return []

    results: list[ValidationResult] = []
    for fp in cypher_files:
        query_name = fp.stem
        cypher_text = fp.read_text(encoding="utf-8").strip()
        if not cypher_text:
            logger.warning("empty_cypher_file", file=fp.name)
            continue

        statements = _split_statements(cypher_text)
        if not statements:
            logger.warning("empty_cypher_file", file=fp.name)
            continue

        logger.info(
            "validation_running",
            query=query_name,
            statements=len(statements),
            timeout_seconds=timeout_seconds,
        )

        # Each top-level statement gets its own bounded execution — a
        # multi-statement file (da#319) is N independent queries as far as the
        # driver and the da#259 timeout budget are concerned, not one.
        block_rows: list[list[dict[str, Any]]] = []
        timeout_detail: str | None = None
        hard_failure = False
        for block_idx, statement in enumerate(statements, start=1):
            try:
                block_rows.append(_execute_read_bounded(client, statement, timeout_seconds))
            except _ValidationTimeout as exc:
                logger.warning(
                    "validation_timeout",
                    query=query_name,
                    block=block_idx,
                    of=len(statements),
                    timeout_seconds=timeout_seconds,
                    detail=str(exc),
                )
                timeout_detail = (
                    f"block {block_idx}/{len(statements)} timed out after "
                    f"{timeout_seconds:.0f}s — downgraded to warning "
                    f"(load succeeded; re-run this check manually, da#259)"
                )
                break
            except Exception:
                logger.exception(
                    "validation_query_failed", query=query_name, block=block_idx, of=len(statements)
                )
                hard_failure = True
                break

        if timeout_detail is not None:
            results.append(
                ValidationResult(
                    query_name, passed=False, details=timeout_detail, row_count=0, warning=True
                )
            )
            continue
        if hard_failure:
            results.append(
                ValidationResult(
                    query_name, passed=False, details="query execution failed", row_count=0
                )
            )
            continue

        # Single-statement files (the pre-da#319 majority) classify on the raw
        # rows exactly as before. Multi-statement files have no single flat
        # row-set, so the classifier gets one per-block summary dict instead
        # (see _classify_informational_inventory).
        if len(statements) == 1:
            rows_for_classifier: list[dict[str, Any]] = block_rows[0]
        else:
            rows_for_classifier = [
                {"block": idx, "row_count": len(rows)}
                for idx, rows in enumerate(block_rows, start=1)
            ]

        result = _classify(query_name, rows_for_classifier, deviation_threshold=deviation_threshold)
        logger.info(
            "validation_complete",
            query=query_name,
            status=result.status,
            rows=result.row_count,
            details=result.details,
        )
        results.append(result)

    return results
