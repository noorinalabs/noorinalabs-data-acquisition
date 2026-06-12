"""Tests for the Itqan acquire downloader (da#92a) — no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.acquire import itqan

_MANIFEST = {
    "files": [
        {"name": "by_name.json", "type": "name_index"},
        {"name": "profiles_reliable.json", "type": "profiles", "grade": "reliable", "count": 2},
        {"name": "profiles_weak.json", "type": "profiles", "grade": "weak", "count": 1},
    ],
    "total_profiles": 3,
    "version": "1.20",
}


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace download_file with a local-writing stub; record requested URLs."""
    requested: list[str] = []

    def _stub(url: str, dest: Path, **_kwargs: object) -> Path:
        requested.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name == "manifest.json":
            dest.write_text(json.dumps(_MANIFEST), encoding="utf-8")
        else:
            dest.write_text(json.dumps({"1": {"id": 1, "full_name": "x"}}), encoding="utf-8")
        return dest

    monkeypatch.setattr(itqan, "download_file", _stub)
    monkeypatch.setattr(itqan, "emit_raw_new_for_manifest", lambda **_kw: None)
    return requested


def test_downloads_only_profile_buckets(fake_download: list[str], tmp_path: Path) -> None:
    dest = itqan.run(tmp_path / "raw")

    assert dest == tmp_path / "raw" / "itqan"
    # Manifest + the two profile buckets fetched; the name_index is NOT (it is #94).
    assert any(u.endswith("manifest.json") for u in fake_download)
    assert any(u.endswith("profiles_reliable.json") for u in fake_download)
    assert any(u.endswith("profiles_weak.json") for u in fake_download)
    assert not any(u.endswith("by_name.json") for u in fake_download)

    # A local manifest.json (write_manifest output) lists the fetched files.
    assert (dest / "manifest.json").exists()
    assert (dest / "profiles_reliable.json").exists()


def test_raises_when_no_profile_buckets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _stub(url: str, dest: Path, **_kwargs: object) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"files": [], "total_profiles": 0}), encoding="utf-8")
        return dest

    monkeypatch.setattr(itqan, "download_file", _stub)
    monkeypatch.setattr(itqan, "emit_raw_new_for_manifest", lambda **_kw: None)

    with pytest.raises(AssertionError, match="no profile buckets"):
        itqan.run(tmp_path / "raw")
