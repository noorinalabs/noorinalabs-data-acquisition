"""Phase 1: Downloaders for hadith data sources.

This package contains source-specific downloaders that fetch raw data from
CSV files, JSON APIs, Kaggle datasets, and Git repositories into ``data/raw/``.
Shared HTTP/download/clone utilities live in ``base.py``.

The set of sources and their run order is the single registry in
:mod:`src.adapters` (epic da#81); :func:`run_all` just iterates it, so this module
and :mod:`src.parse` can no longer drift on which sources exist.
"""

from __future__ import annotations

from pathlib import Path

from src.adapters import SOURCE_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_all(raw_dir: Path) -> dict[str, Path | None]:
    """Run all downloaders. Continue on failure. Return dict of source -> path."""
    results: dict[str, Path | None] = {}
    for adapter in SOURCE_REGISTRY:
        name = adapter.slug
        if not adapter.active:
            # Inactive sources (e.g. open_hadith — a confirmed duplicate, da#191)
            # are never acquired. The registry row is retained for the coverage
            # invariant only.
            logger.info("acquire_skipped_inactive", source=name)
            continue
        try:
            logger.info("acquiring", source=name)
            path = adapter.acquire(raw_dir)
            results[name] = path
            logger.info("acquired", source=name, path=str(path) if path else "skipped")
        except Exception as exc:  # noqa: BLE001
            logger.error("acquire_failed", source=name, error=str(exc), exc_info=True)
            results[name] = None

    # Summary table
    print("\n=== Acquisition Summary ===")
    for name, path in results.items():
        status = "ok" if path else "FAIL"
        if path and path.exists():
            files = [f for f in path.rglob("*") if f.is_file()]
            file_count = len(files)
            total_size = sum(f.stat().st_size for f in files)
            size_mb = total_size / 1024 / 1024
            print(f"  [{status:4s}] {name:15s}  {file_count:>4d} files  {size_mb:>8.1f} MB")
        else:
            print(f"  [{status:4s}] {name:15s}  --")

    return results
