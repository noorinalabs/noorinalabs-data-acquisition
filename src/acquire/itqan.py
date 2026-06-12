"""Acquire the Itqan narrator (rijal) profile dataset from GitHub.

Itqan (github ``R3GENESI5/Itqan``) is the largest open-source narrator
database: 115,735 narrator profiles drawn from 22 classical rijal texts, 72.6%
of them carrying a jarh-wa-ta'dil grade. The profiles live under
``app/data/rijal/`` as one JSON file per grade bucket (``profiles_<grade>.json``)
plus a ``manifest.json`` that enumerates the buckets and their counts.

This downloader fetches **only the narrator-profile buckets** (the highest-value
slice — da#92a). The sibling artefacts are intentionally left for the other two
Itqan PRs in the pre-split (per epic da#81):

* ``teachers`` / ``students`` id lists inside each profile → isnad chains (#93).
* ``by_name.json`` (the name-variant index) and the per-profile ``namings`` →
  name variants (#94).

Reachability: served from ``raw.githubusercontent.com`` over HTTPS — no auth,
no API key. The seven profile buckets total ~150 MB.

Licensing: the upstream repo ships **no LICENSE file** (all-rights-reserved by
default). Acquisition here is for research/analysis; redistribution or
production publication of the raw corpus needs an explicit usage grant from the
upstream author (Ali Bin Shahid) — flagged for owner confirmation, do NOT treat
as cleared for redistribution.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.acquire.base import download_file, ensure_dir, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

RIJAL_BASE_URL = "https://raw.githubusercontent.com/R3GENESI5/Itqan/master/app/data/rijal"
MANIFEST_URL = f"{RIJAL_BASE_URL}/manifest.json"


def run(raw_dir: Path) -> Path:
    """Download the Itqan rijal narrator-profile buckets.

    Idempotent: ``download_file`` skips a non-empty existing file, so re-running
    only fetches what is missing (the manifest itself is always refreshed so a
    new upstream version is noticed).
    """
    dest = ensure_dir(raw_dir / "itqan")

    # Manifest first — it is the source of truth for which buckets exist.
    manifest_path = download_file(MANIFEST_URL, dest / "manifest.json", overwrite=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    profile_entries = [e for e in manifest.get("files", []) if e.get("type") == "profiles"]
    if not profile_entries:
        msg = f"Itqan manifest has no profile buckets: {manifest_path}"
        raise AssertionError(msg)

    downloaded: list[Path] = [manifest_path]
    expected_total = 0
    for entry in profile_entries:
        name = entry["name"]
        expected_total += int(entry.get("count", 0))
        path = download_file(f"{RIJAL_BASE_URL}/{name}", dest / name)
        downloaded.append(path)

    logger.info(
        "itqan_acquired",
        buckets=len(profile_entries),
        expected_profiles=expected_total,
        manifest_total=manifest.get("total_profiles"),
        version=manifest.get("version"),
    )

    write_manifest(dest, downloaded)
    emit_raw_new_for_manifest(source="itqan", local_dir=dest, files=downloaded)
    return dest
