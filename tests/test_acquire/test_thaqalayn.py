"""Tests for the Thaqalayn (Shia) acquire adapter.

da#85 — closing the acquire-side test gap (only ``base`` and ``sunnah_scraper``
had one before). The network/clone is mocked: ``clone_repo`` is patched to
materialise fake book JSONs so the GitHub-clone, idempotent-skip, and
under-minimum branches of ``run`` are exercised without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.acquire.thaqalayn import MIN_EXPECTED_BOOKS, run


def _fake_clone(_url: str, dest: Path, **_kwargs: object) -> Path:
    """Stand-in for ``clone_repo``: materialise MIN_EXPECTED_BOOKS JSON files."""
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(MIN_EXPECTED_BOOKS):
        (dest / f"book_{i}.json").write_text(
            json.dumps({"bookName": f"Book {i}", "data": []}), encoding="utf-8"
        )
    return dest


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

        # Manifest lists the cloned JSONs.
        manifest = json.loads((dest / "manifest.json").read_text())
        assert len(manifest) >= MIN_EXPECTED_BOOKS

    @patch("src.acquire.thaqalayn.emit_raw_new_for_manifest")
    @patch("src.acquire.thaqalayn.clone_repo", side_effect=_fake_clone)
    def test_idempotent_skips_when_already_acquired(
        self, mock_clone: object, mock_emit: object, tmp_path: Path
    ) -> None:
        """Pre-existing ``book_*.json`` short-circuits the clone (no network)."""
        dest = tmp_path / "raw" / "thaqalayn"
        dest.mkdir(parents=True)
        for i in range(MIN_EXPECTED_BOOKS):
            (dest / f"book_{i}.json").write_text("{}", encoding="utf-8")

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
        """A clone yielding fewer than MIN_EXPECTED_BOOKS JSONs fails loudly."""

        def _sparse_clone(_url: str, clone_dest: Path, **_kwargs: object) -> Path:
            clone_dest.mkdir(parents=True, exist_ok=True)
            (clone_dest / "book_0.json").write_text("{}", encoding="utf-8")
            return clone_dest

        mock_clone.side_effect = _sparse_clone  # type: ignore[attr-defined]

        with pytest.raises(AssertionError, match="JSON files"):
            run(tmp_path / "raw")

        mock_emit.assert_not_called()  # type: ignore[attr-defined]
