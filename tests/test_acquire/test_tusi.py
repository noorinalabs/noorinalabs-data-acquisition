"""Tests for the ThaqalaynData (Tahdhib + al-Istibsar) acquire adapter.

The network clone is mocked: ``clone_repo`` is patched to materialise the two
target Book directories (or to omit one, exercising the structural-guard branch)
without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.acquire.tusi import TARGET_BOOK_SLUGS, run


def _fake_clone(_url: str, dest: Path, **_kwargs: object) -> Path:
    """Stand-in for ``clone_repo``: materialise a verse_detail JSON per target Book."""
    for slug in TARGET_BOOK_SLUGS:
        book_dir = dest / "books" / slug / "1" / "1"
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "1.json").write_text(
            json.dumps({"kind": "verse_detail", "data": {}}), encoding="utf-8"
        )
    return dest


def _fake_clone_missing_book(_url: str, dest: Path, **_kwargs: object) -> Path:
    """Stand-in that materialises only ONE Book — the other is missing."""
    book_dir = dest / "books" / TARGET_BOOK_SLUGS[0] / "1" / "1"
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "1.json").write_text(json.dumps({"kind": "verse_detail"}), encoding="utf-8")
    return dest


class TestThaqalaynDataAcquire:
    @patch("src.acquire.tusi.emit_raw_new_for_manifest")
    @patch("src.acquire.tusi.clone_repo", side_effect=_fake_clone)
    def test_clones_and_writes_manifest(
        self, mock_clone: object, mock_emit: object, tmp_path: Path
    ) -> None:
        raw_dir = tmp_path / "raw"

        dest = run(raw_dir)

        assert dest == raw_dir / "tusi"
        assert (dest / "manifest.json").exists()
        mock_clone.assert_called_once()  # type: ignore[attr-defined]
        mock_emit.assert_called_once()  # type: ignore[attr-defined]
        # Manifest lists one JSON per target Book.
        manifest = json.loads((dest / "manifest.json").read_text())
        assert len(manifest) == len(TARGET_BOOK_SLUGS)

    @patch("src.acquire.tusi.emit_raw_new_for_manifest")
    @patch("src.acquire.tusi.clone_repo", side_effect=_fake_clone_missing_book)
    def test_missing_book_directory_raises(
        self, _mock_clone: object, _mock_emit: object, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="missing expected Book directories"):
            run(tmp_path / "raw")
