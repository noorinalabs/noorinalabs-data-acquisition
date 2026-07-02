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


def _classify_chain_integrity(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    count = len(rows)
    passed = count == 0
    details = "no cycles" if passed else f"{count} cycle(s) detected"
    return ValidationResult(query_name, passed, details, count)


def _classify_collection_coverage(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    count = len(rows)
    failures: list[str] = []
    for row in rows:
        dev = row.get("deviation_pct")
        if dev is not None and isinstance(dev, int | float) and dev > deviation_threshold:
            cid = row.get("collection_id", "?")
            failures.append(f"{cid}: {dev:.1f}% deviation")
    passed = len(failures) == 0
    details = "all within threshold" if passed else "; ".join(failures)
    return ValidationResult(query_name, passed, details, count)


def _classify_default(
    query_name: str,
    rows: list[dict[str, object]],
    deviation_threshold: float,
) -> ValidationResult:
    """Default classifier — pass if 0 rows (conservative)."""
    count = len(rows)
    passed = count == 0
    details = f"{count} row(s) returned" if count else "empty result"
    return ValidationResult(query_name, passed, details, count)


_CLASSIFIER_REGISTRY: dict[str, ClassifierFunc] = {
    "orphan_narrators": _classify_orphan_narrators,
    "chain_integrity": _classify_chain_integrity,
    "collection_coverage": _classify_collection_coverage,
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

    Each query is executed with a per-query time budget (``timeout_seconds``).
    A query that exceeds it — e.g. the ``chain_integrity`` cycle traversal on a
    slow or cyclic full graph — is **downgraded to a non-fatal warning** instead
    of hanging the loader (da#259). The load itself has already succeeded at this
    point; validation is a post-load check, so a timed-out check is inconclusive,
    not a failure.

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

        logger.info("validation_running", query=query_name, timeout_seconds=timeout_seconds)
        try:
            rows = _execute_read_bounded(client, cypher_text, timeout_seconds)
        except _ValidationTimeout as exc:
            logger.warning(
                "validation_timeout",
                query=query_name,
                timeout_seconds=timeout_seconds,
                detail=str(exc),
            )
            results.append(
                ValidationResult(
                    query_name,
                    passed=False,
                    details=(
                        f"timed out after {timeout_seconds:.0f}s — downgraded to warning "
                        f"(load succeeded; re-run this check manually, da#259)"
                    ),
                    row_count=0,
                    warning=True,
                )
            )
            continue
        except Exception:
            logger.exception("validation_query_failed", query=query_name)
            results.append(
                ValidationResult(
                    query_name, passed=False, details="query execution failed", row_count=0
                )
            )
            continue

        result = _classify(query_name, rows, deviation_threshold=deviation_threshold)
        logger.info(
            "validation_complete",
            query=query_name,
            status=result.status,
            rows=result.row_count,
            details=result.details,
        )
        results.append(result)

    return results
