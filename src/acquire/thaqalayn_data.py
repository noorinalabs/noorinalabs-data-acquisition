"""Acquire the Shia Four Books *Tahdhib al-Ahkam* + *al-Istibsar* from ThaqalaynData.

The Four Books are split across our Shia sources by *upstream*: the existing
``thaqalayn`` adapter (:mod:`src.acquire.thaqalayn`) clones
``MohammedArab1/ThaqalaynAPI`` — a weekly *website scrape* of thaqalayn.net that
carries al-Kafi + Man-la-yahduruh-al-Faqih but NOT the other two Books (verified
da#182). The two missing Books live in the project's ORIGINAL data backend,
``narmafraz/ThaqalaynData`` — the CC0 dataset that thaqalayn.net itself renders —
which carries all four Books with the real Arabic source text.

This downloader clones that repo (idempotent — a populated clone is reused). It is
a DIFFERENT upstream, schema, and license from ``thaqalayn`` (hence its own
``thaqalayn_data`` corpus), so the two never share a parser or a corpus
namespace. :mod:`src.parse.thaqalayn_data` parses the clone.

Licence: ``narmafraz/ThaqalaynData`` is **CC0 1.0 Universal** (public-domain
dedication; verified LICENSE file) — the cleanest provenance of any source we
use. NB the dataset's *non-Arabic translations* are AI-generated
(``verse.ai``, ``model=pipeline_v4``); da#182 loads the **real Arabic only** and
deliberately omits the machine translations (see the parser).
"""

from __future__ import annotations

from pathlib import Path

from src.acquire.base import clone_repo, ensure_dir, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

THAQALAYN_DATA_GITHUB_URL = "https://github.com/narmafraz/ThaqalaynData.git"

# The two Books da#182 sources from this upstream (the ``books/<slug>/`` dirs).
TARGET_BOOK_SLUGS: tuple[str, ...] = ("tahdhib-al-ahkam", "al-istibsar")


def _book_dir(clone_root: Path, slug: str) -> Path:
    """Path to one Book's directory inside a ThaqalaynData clone."""
    return clone_root / "books" / slug


def run(raw_dir: Path) -> Path:
    """Clone ThaqalaynData and return the ``thaqalayn_data`` raw directory.

    Idempotent: an existing clone that already contains both target Book
    directories is reused (``clone_repo`` itself also skips a populated dest).
    Raises if, after cloning, either Book directory is missing — a structural
    upstream change we must not parse around silently.
    """
    dest = ensure_dir(raw_dir / "thaqalayn_data")
    clone_root = dest / "clone"

    clone_repo(THAQALAYN_DATA_GITHUB_URL, clone_root)

    missing = [slug for slug in TARGET_BOOK_SLUGS if not _book_dir(clone_root, slug).is_dir()]
    if missing:
        msg = (
            f"ThaqalaynData clone is missing expected Book directories {missing} under "
            f"{clone_root / 'books'} — upstream layout changed; refusing to parse around it."
        )
        raise FileNotFoundError(msg)

    # Manifest/event over the two Books' JSON files only (not the whole repo).
    book_files = [
        path
        for slug in TARGET_BOOK_SLUGS
        for path in sorted(_book_dir(clone_root, slug).rglob("*.json"))
    ]
    logger.info("thaqalayn_data_acquired", books=len(TARGET_BOOK_SLUGS), files=len(book_files))

    write_manifest(dest, book_files)
    emit_raw_new_for_manifest(source="thaqalayn_data", local_dir=dest, files=book_files)
    return dest
