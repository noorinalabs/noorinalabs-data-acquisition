"""Acquire the Multi-IsnadSet (MIS) dataset from Mendeley Data.

MIS (``Multi-IsnadSet``, Farooqi et al., *Data in Brief* 2024 —
DOI ``10.17632/gzprcr93zn.2``) models **Sahih Muslim** as a directed narrator
graph in which a single hadith carries its *multiple* isnad chains explicitly
(2,092 narrator nodes / 77,797 Sanad-Hadith edges). It is the dedicated
multi-isnad source for the platform: where most corpora give one chain per
hadith, MIS preserves the parallel asanid (the ``ح`` / *tahwil* split) that make
Sahih Muslim valuable for chain modelling.

The dataset ships as two Excel workbooks under the Mendeley file set:

* ``Hadith_SahihMuslim_CoreInfo.xlsx``               — one row per hadith
  (book, hadith number, matn, narrator sequence, isnad count).
* ``Hadith_SahihMuslim_DetailsInfo_Sanad_Narrators.xlsx`` — one row per
  (hadith, sanad, narrator) position — the data the parser walks into per-sanad
  chains.

Reachability: ``data.mendeley.com`` serves the file set over HTTPS with **no
auth / no API key** — the same keyless-public mechanism this repo already proves
for ``sanadset`` (Mendeley ``5xth87zwb5``). The file ids are resolved at runtime
from the dataset files API rather than hardcoded, so a new dataset version is
picked up without a code change.

Licensing: Mendeley Data's default license for this dataset is **CC BY 4.0**
(attribution) — redistribution-safe with credit to the depositors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.acquire.base import download_file, ensure_dir, fetch_json, write_manifest
from src.messaging import emit_raw_new_for_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Mendeley Data — Multi-IsnadSet (DOI: 10.17632/gzprcr93zn.2)
_MENDELEY_DATASET_ID = "gzprcr93zn"
_MENDELEY_VERSION = 2
_MENDELEY_FILES_API = f"https://data.mendeley.com/api/datasets/{_MENDELEY_DATASET_ID}/files?version={_MENDELEY_VERSION}"

# We only want the spreadsheet workbooks; the file set may also carry PDFs / docs.
_WANTED_SUFFIXES = (".xlsx", ".xls")


def _resolve_files() -> list[dict[str, Any]]:
    """Return the Mendeley file-set entries for the wanted spreadsheet workbooks.

    Each entry is normalized to ``{"name", "url", "sha256"}``. The Mendeley files
    API returns objects shaped ``{"filename", "content_details": {"download_url",
    "sha256_hash", ...}}``; we tolerate a couple of historical key spellings.
    """
    raw = fetch_json(_MENDELEY_FILES_API)
    entries = raw if isinstance(raw, list) else raw.get("results") or raw.get("files") or []

    wanted: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("filename") or entry.get("name") or ""
        if not name.lower().endswith(_WANTED_SUFFIXES):
            continue
        details = entry.get("content_details") or entry
        url = details.get("download_url") or details.get("url")
        if not url:
            continue
        wanted.append(
            {
                "name": name,
                "url": url,
                "sha256": details.get("sha256_hash") or details.get("sha256"),
            }
        )
    return wanted


def run(raw_dir: Path) -> Path:
    """Download the MIS Excel workbooks into ``raw_dir/mis``.

    Idempotent: ``download_file`` skips a non-empty existing workbook, so a
    re-run only fetches what is missing. Fails fast if the file set carries no
    spreadsheet workbook (a dataset-shape change we want to notice loudly).
    """
    dest = ensure_dir(raw_dir / "mis")

    files = _resolve_files()
    if not files:
        msg = (
            f"MIS Mendeley file set ({_MENDELEY_DATASET_ID} v{_MENDELEY_VERSION}) "
            "exposed no .xlsx workbook"
        )
        raise AssertionError(msg)

    downloaded: list[Path] = []
    for entry in files:
        path = download_file(
            entry["url"],
            dest / entry["name"],
            expected_sha256=entry["sha256"],
            timeout=600.0,
        )
        downloaded.append(path)

    logger.info("mis_acquired", workbooks=len(downloaded), names=[p.name for p in downloaded])

    write_manifest(dest, downloaded)
    emit_raw_new_for_manifest(source="mis", local_dir=dest, files=downloaded)
    return dest
