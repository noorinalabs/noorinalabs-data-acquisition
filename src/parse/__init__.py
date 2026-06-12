"""Phase 1: Parsers producing normalized Parquet from raw data.

The set of sources and their run order is the single registry in
:mod:`src.adapters` (epic da#81); :func:`run_all` iterates it, so the parse and
acquire orchestrators can no longer drift on which sources exist.
"""

from __future__ import annotations

from pathlib import Path

from src.adapters import SOURCE_REGISTRY, ParseOutput
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_output(result: ParseOutput) -> list[Path]:
    """Normalize parser return values to a flat list of Paths."""
    if isinstance(result, dict):
        return list(result.values())
    if isinstance(result, tuple):
        return list(result)
    if isinstance(result, list):
        return result
    return [result]


def run_all(raw_dir: Path, staging_dir: Path) -> dict[str, list[Path]]:
    """Run all parsers. Continue on failure. Return dict of source -> output files."""
    results: dict[str, list[Path]] = {}
    for adapter in SOURCE_REGISTRY:
        name = adapter.slug
        try:
            logger.info("parsing", source=name)
            output_files = _normalize_output(adapter.parse(raw_dir, staging_dir))
            results[name] = output_files
            logger.info("parsed", source=name, files=len(output_files))
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
    return results
