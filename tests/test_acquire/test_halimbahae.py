"""Tests for the halimbahae/Hadith acquire downloader (da#96) — no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.acquire import halimbahae

_BOOKS = [
    "Maliks_Muwataa/maliks_muwataa",
    "Musnad_Ahmad_Ibn-Hanbal/musnad_ahmad_ibn-hanbal",
    "Sahih_Al-Bukhari/sahih_al-bukhari",
    "Sahih_Muslim/sahih_muslim",
    "Sunan_Abu-Dawud/sunan_abu-dawud",
    "Sunan_Al-Darimi/sunan_al-darimi",
    "Sunan_Al-Nasai/sunan_al-nasai",
    "Sunan_Al-Tirmidhi/sunan_al-tirmidhi",
    "Sunan_Ibn-Maja/sunan_ibn-maja",
]


def _fake_clone(books: list[str]):
    """Build a clone_repo stub that materialises *books*' diacritised CSVs."""

    def _clone(_url: str, dest: Path, **_kw: object) -> Path:
        for slug in books:
            csv = dest / f"{slug}_ahadith_mushakkala_mufassala.utf8.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text('"1","حَدَّثَنَا"\n', encoding="utf-8")
            # The plain (non-mushakkala) sibling must be ignored by the count.
            plain = dest / f"{slug}_ahadith.utf8.csv"
            plain.write_text('"1","حدثنا"\n', encoding="utf-8")
        return dest

    return _clone


def test_acquires_when_all_books_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(halimbahae, "clone_repo", _fake_clone(_BOOKS))
    monkeypatch.setattr(halimbahae, "emit_raw_new_for_manifest", lambda **_kw: None)

    dest = halimbahae.run(tmp_path / "raw")

    assert dest == tmp_path / "raw" / "halimbahae"
    # A local manifest.json (write_manifest output) lists the diacritised CSVs.
    assert (dest / "manifest.json").exists()
    diacritics_csvs = [p for p in dest.rglob("*.csv") if "mushakkala" in p.name.lower()]
    assert len(diacritics_csvs) == 9


def test_raises_when_too_few_books(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Only 3 books present — below the 9-book floor.
    monkeypatch.setattr(halimbahae, "clone_repo", _fake_clone(_BOOKS[:3]))
    monkeypatch.setattr(halimbahae, "emit_raw_new_for_manifest", lambda **_kw: None)

    with pytest.raises(AssertionError, match="mushakkala"):
        halimbahae.run(tmp_path / "raw")
