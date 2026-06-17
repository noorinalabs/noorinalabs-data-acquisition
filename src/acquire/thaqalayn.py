"""Acquire hadith data from the Thaqalayn GitHub repository.

Fetches Shia hadith collections by cloning the MohammedArab1/ThaqalaynAPI
GitHub repository. The Thaqalayn REST API v2 was removed circa early 2026
when the website was rebuilt with Next.js — the GitHub repo is now the
primary acquisition method.

The repo ships the dataset multiple times (V1 + V2 trees, each with a giant
``allBooks.json`` aggregate) plus build/index files. Acquisition has version/
scope discipline: it counts and emits only the **canonical per-book files**
(``V2/ThaqalaynData/<n>.json``) via :func:`src.parse.thaqalayn.book_json_files`,
the single selector shared with the parser, so the manifest + ``raw_new`` event
never reference the aggregates or repo config that the old ``rglob`` swept in
(da#175).
"""

from __future__ import annotations

from pathlib import Path

from src.acquire.base import clone_repo, ensure_dir, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.parse.thaqalayn import book_json_files
from src.utils.logging import get_logger

logger = get_logger(__name__)

THAQALAYN_GITHUB_URL = "https://github.com/MohammedArab1/ThaqalaynAPI.git"
MIN_EXPECTED_BOOKS = 15


def run(raw_dir: Path) -> Path:
    """Download Thaqalayn data via GitHub clone (idempotent)."""
    dest = ensure_dir(raw_dir / "thaqalayn")

    # Idempotent skip: a prior clone already has the canonical per-book files.
    existing = book_json_files(dest)
    if len(existing) >= MIN_EXPECTED_BOOKS:
        logger.info("thaqalayn_skipped", reason="already_acquired", file_count=len(existing))
        write_manifest(dest, existing)
        emit_raw_new_for_manifest(source="thaqalayn", local_dir=dest, files=existing)
        return dest

    clone_repo(THAQALAYN_GITHUB_URL, dest / "github_clone")
    book_files = book_json_files(dest)
    logger.info("thaqalayn_github_clone", book_count=len(book_files))

    if len(book_files) < MIN_EXPECTED_BOOKS:
        msg = (
            f"Expected >= {MIN_EXPECTED_BOOKS} canonical per-book JSON files under "
            f"V2/ThaqalaynData, found {len(book_files)}"
        )
        raise AssertionError(msg)

    write_manifest(dest, book_files)
    emit_raw_new_for_manifest(source="thaqalayn", local_dir=dest, files=book_files)
    logger.info("thaqalayn_acquired", total_files=len(book_files))
    return dest
