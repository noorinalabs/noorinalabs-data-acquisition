"""Tests for the Thaqalayn (Shia) acquire adapter.

The network clone is mocked: ``clone_repo`` is patched to materialise fake
per-book JSONs under the real ``V2/ThaqalaynData/<n>.json`` layout, so the
clone, idempotent-skip, and under-minimum branches of ``run`` are exercised
without touching the network. Scope discipline (da#175): the adapter counts and
emits only the canonical per-book files, never the aggregates/config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.acquire.thaqalayn import MIN_EXPECTED_BOOKS, run


def _data_dir(clone_dest: Path) -> Path:
    """The canonical per-book dir inside a clone destination."""
    return clone_dest / "V2" / "ThaqalaynData"


def _write_book_files(data_dir: Path, count: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (data_dir / f"{i}.json").write_text(
            json.dumps([{"id": 1, "bookId": f"Book-{i}", "arabicText": "نص"}], ensure_ascii=False),
            encoding="utf-8",
        )
    # Aggregate/config siblings that must NOT be counted as book files.
    (data_dir / "allBooks.json").write_text("[]", encoding="utf-8")
    (data_dir / "BookNames.json").write_text("[]", encoding="utf-8")


def _fake_clone(_url: str, clone_dest: Path, **_kwargs: object) -> Path:
    """Stand-in for ``clone_repo``: materialise MIN_EXPECTED_BOOKS book files."""
    _write_book_files(_data_dir(clone_dest), MIN_EXPECTED_BOOKS)
    return clone_dest


class TestThaqalaynAcquire:
    @patch("src.acquire.thaqalayn.emit_raw_new_for_manifest")
    @patch("src.acquire.thaqalayn.clone_repo", side_effect=_fake_clone)
    def test_clones_and_writes_manifest(
        self, mock_clone: object, mock_emit: object, tmp_path: Path
    ) -> None:
        """Happy path: clone the repo, then write a manifest + emit the event."""
        raw_dir = tmp_path / "raw"

        dest = run(raw_dir)

        assert dest == raw_dir / "thaqalayn"
        assert (dest / "manifest.json").exists()
        mock_clone.assert_called_once()  # type: ignore[attr-defined]
        mock_emit.assert_called_once()  # type: ignore[attr-defined]

        # Manifest lists exactly the canonical per-book files (not aggregates).
        manifest = json.loads((dest / "manifest.json").read_text())
        assert len(manifest) == MIN_EXPECTED_BOOKS

    @patch("src.acquire.thaqalayn.emit_raw_new_for_manifest")
    @patch("src.acquire.thaqalayn.clone_repo", side_effect=_fake_clone)
    def test_idempotent_skips_when_already_acquired(
        self, mock_clone: object, mock_emit: object, tmp_path: Path
    ) -> None:
        """A pre-existing clone short-circuits the network clone."""
        dest = tmp_path / "raw" / "thaqalayn"
        _write_book_files(_data_dir(dest / "github_clone"), MIN_EXPECTED_BOOKS)

        result = run(tmp_path / "raw")

        assert result == dest
        mock_clone.assert_not_called()  # type: ignore[attr-defined]
        # The skip path still publishes a manifest + raw-new event.
        assert (dest / "manifest.json").exists()
        mock_emit.assert_called_once()  # type: ignore[attr-defined]

    @patch("src.acquire.thaqalayn.emit_raw_new_for_manifest")
    @patch("src.acquire.thaqalayn.clone_repo")
    def test_raises_when_too_few_books(
        self, mock_clone: object, mock_emit: object, tmp_path: Path
    ) -> None:
        """A clone yielding fewer than MIN_EXPECTED_BOOKS book files fails loudly."""

        def _sparse_clone(_url: str, clone_dest: Path, **_kwargs: object) -> Path:
            _write_book_files(_data_dir(clone_dest), 1)
            return clone_dest

        mock_clone.side_effect = _sparse_clone  # type: ignore[attr-defined]

        with pytest.raises(AssertionError, match="per-book JSON files"):
            run(tmp_path / "raw")

        mock_emit.assert_not_called()  # type: ignore[attr-defined]
