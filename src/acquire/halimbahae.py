"""Acquire the halimbahae/Hadith corpus from GitHub.

Shallow-clones halimbahae/Hadith which ships 9 Sunni books (the Six Books plus
Muwatta Malik, Musnad Ahmad and Sunan al-Darimi). Each book directory carries a
plain CSV and a ``*_mushakkala*.utf8.csv`` variant with full Arabic diacritics
(tashkeel) — the parser keys on the diacritised variant.
"""

from __future__ import annotations

from pathlib import Path

from src.acquire.base import clone_repo, ensure_dir, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

REPO_URL = "https://github.com/halimbahae/Hadith.git"
# Nine book directories, one diacritised CSV each.
MIN_EXPECTED_DIACRITICS_CSVS = 9


def run(raw_dir: Path) -> Path:
    """Clone halimbahae/Hadith and validate the diacritised CSVs exist.

    Idempotent -- skips the clone if the destination is already populated.
    """
    dest = ensure_dir(raw_dir / "halimbahae")
    clone_repo(REPO_URL, dest)

    # The parser consumes only the diacritised (``mushakkala``) CSVs; assert we
    # have at least one per book before declaring the source acquired.
    diacritics_csvs = [p for p in dest.rglob("*.csv") if "mushakkala" in p.name.lower()]
    if len(diacritics_csvs) < MIN_EXPECTED_DIACRITICS_CSVS:
        msg = (
            f"Expected >={MIN_EXPECTED_DIACRITICS_CSVS} diacritised (mushakkala) CSVs, "
            f"found {len(diacritics_csvs)}"
        )
        raise AssertionError(msg)

    logger.info("halimbahae_acquired", diacritics_csv_count=len(diacritics_csvs))
    write_manifest(dest, diacritics_csvs)
    emit_raw_new_for_manifest(source="halimbahae", local_dir=dest, files=diacritics_csvs)
    return dest
