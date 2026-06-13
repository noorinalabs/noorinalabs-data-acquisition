"""Tests for the Sunnah.com API acquire downloader (da#91).

Two legs:

* **Keyless graceful-skip** (always runs, no network) — locks the contract CI
  relies on: with ``SUNNAH_API_KEY`` absent, ``run()`` returns ``None`` and
  writes nothing, so the source is *adapter-ready* without failing the build.
* **Live API smoke** (skip-guarded behind ``SUNNAH_API_KEY``) — exercises the
  real ``api.sunnah.com/v1`` leg + the real parser on live data, but only when a
  key is present.  The key was never granted (403 keyless, **da#71**), so this
  leg ``skip``s off-key and lights up once the owner provisions the key.  See
  the ``sunnah`` row in ``src/adapters.py`` (``reachable=False``) and
  ``docs/adapters.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.acquire import sunnah_api
from src.config import get_settings
from src.parse.schemas import COLLECTION_SCHEMA
from src.parse.sunnah_api import run as parse_run

_HAS_KEY = bool(os.getenv("SUNNAH_API_KEY"))
_SKIP_REASON = "da#71: SUNNAH_API_KEY not granted — live Sunnah.com API leg skip-guarded"


class TestSunnahApiKeylessSkip:
    """The keyless graceful-skip path — always runs, never touches the network."""

    def test_run_returns_none_without_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure the key is absent regardless of the ambient environment, then
        # drop the cached Settings singleton so the absence is observed.
        monkeypatch.delenv("SUNNAH_API_KEY", raising=False)
        get_settings.cache_clear()

        raw_dir = tmp_path / "raw"
        result = sunnah_api.run(raw_dir)

        # Graceful skip: no output dir, no network, no writes.
        assert result is None
        assert not (raw_dir / "sunnah").exists()


@pytest.mark.skipif(not _HAS_KEY, reason=_SKIP_REASON)
class TestSunnahApiLive:
    """The live API leg — runs only when a real SUNNAH_API_KEY is present."""

    def test_live_collections_acquire_and_parse(self, tmp_path: Path) -> None:
        """Fetch the real collections endpoint and parse it to conforming Parquet.

        Bounded on purpose: this hits ``GET /collections`` (a few dozen rows)
        rather than calling ``sunnah_api.run()``, which would pull every hadith
        of every collection.  The full production load is the pipeline's job
        (``make acquire`` / ``make parse``); this leg just proves the key
        authenticates and real data flows through the parser cleanly.
        """
        # Pick up the real key from the environment.
        get_settings.cache_clear()
        api_key = get_settings().sunnah_api_key
        assert api_key, "live leg requires SUNNAH_API_KEY"

        from src.acquire.base import fetch_json_paginated

        collections = fetch_json_paginated(
            f"{sunnah_api.BASE_URL}/collections",
            headers={"X-API-Key": api_key},
            limit=50,
        )
        assert collections, "live /collections returned no data"

        # Lay the fetched data out the way the parser expects, then run the real
        # parser.  No ``*_hadiths.json`` files are written, so the parser emits
        # only the collections table — enough to prove the live→parse contract.
        raw_dir = tmp_path / "raw"
        staging_dir = tmp_path / "staging"
        sunnah_dir = raw_dir / "sunnah"
        sunnah_dir.mkdir(parents=True)
        (sunnah_dir / "collections.json").write_text(
            json.dumps(collections, ensure_ascii=False), encoding="utf-8"
        )

        output_files = parse_run(raw_dir, staging_dir)

        collections_parquet = staging_dir / "collections_sunnah.parquet"
        assert collections_parquet in output_files
        table = pq.read_table(collections_parquet)
        assert table.schema == COLLECTION_SCHEMA
        assert table.num_rows > 0
