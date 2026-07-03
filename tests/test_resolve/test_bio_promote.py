"""Tests for the bio-direct canonical promoter (da#92a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.parse.schemas import NARRATOR_ALIAS_SCHEMA, NARRATOR_BIO_SCHEMA
from src.resolve.bio_promote import promote_bios_to_canonical
from src.utils.arabic import normalize_arabic


def _write_bios(staging: Path, suffix: str, rows: list[dict[str, Any]]) -> Path:
    """Write a narrators_bio_<suffix>.parquet with schema defaults filled in."""
    full = []
    for r in rows:
        base = {f.name: None for f in NARRATOR_BIO_SCHEMA}
        base.update(r)
        full.append(base)
    arrays = {f.name: [r[f.name] for r in full] for f in NARRATOR_BIO_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    path = staging / f"narrators_bio_{suffix}.parquet"
    pq.write_table(table, path)
    return path


def _write_aliases(staging: Path, suffix: str, rows: list[dict[str, Any]]) -> Path:
    """Write a narrator_aliases_<suffix>.parquet."""
    arrays = {f.name: [r[f.name] for r in rows] for f in NARRATOR_ALIAS_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_ALIAS_SCHEMA)
    path = staging / f"narrator_aliases_{suffix}.parquet"
    pq.write_table(table, path)
    return path


def test_promotes_bios_to_canonical_with_nar_ids(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "سعيد بن سماك"
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": normalize_arabic(name),
                "trustworthiness": "matruk",
                "external_id": "320",
            }
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    table = pq.read_table(path)
    rows = table.to_pylist()
    assert len(rows) == 1
    rec = rows[0]
    assert rec["canonical_id"].startswith("nar:")
    assert rec["canonical_id"] == make_canonical_id(normalize_arabic(name))
    assert rec["trustworthiness"] == "matruk"
    assert rec["external_id"] == "320"
    assert rec["source_ids"] == ["itqan:320"]
    assert rec["mention_count"] == 0


def test_bio_promote_tags_sect_and_corpus(tmp_path: Path) -> None:
    """da#103: promoted bios carry source_corpus/_corpora + derived sect_affiliation."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "فاطمة بنت محمد"
    _write_bios(
        staging,
        "muhaddithat",
        [
            {
                "bio_id": "muhaddithat:1",
                "source": "muhaddithat",
                "name_ar": name,
                "name_ar_normalized": normalize_arabic(name),
            }
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rec = pq.read_table(path).to_pylist()[0]
    assert rec["source_corpora"] == ["muhaddithat"]
    assert rec["source_corpus"] == "muhaddithat"
    # muhaddithat is cross-tradition (spans both sects, no per-narrator sect in
    # the source) → a muhaddithat-only narrator is neutral, not a sunni guess
    # (da#90).
    assert rec["sect_affiliation"] == "neutral"


def test_same_name_dedups_to_one_canonical(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "عمر بن حفص"
    norm = normalize_arabic(name)
    # The second bio carries an int death year; back-fill must keep it an int
    # (regression: it was previously stringified, breaking the int32 column).
    _write_bios(
        staging,
        "itqan",
        [
            {"bio_id": "itqan:1", "source": "itqan", "name_ar": name, "name_ar_normalized": norm},
            {
                "bio_id": "itqan:2",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": norm,
                "death_year_ah": 231,
            },
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 1
    assert sorted(rows[0]["source_ids"]) == ["itqan:1", "itqan:2"]
    assert rows[0]["death_year_ah"] == 231


def test_source_filter(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:1", "source": "itqan", "name_ar": "أ", "name_ar_normalized": "ا"}],
    )
    _write_bios(
        staging,
        "muhaddithat",
        [{"bio_id": "m:1", "source": "muhaddithat", "name_ar": "ب", "name_ar_normalized": "ب"}],
    )
    path = promote_bios_to_canonical(staging, out_dir, sources={"itqan"})
    assert path is not None
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["source_ids"] == ["itqan:1"]


def test_merges_into_existing_canonical(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    # Pre-existing canonical (as if from the disambiguator) with a different id.
    from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA

    existing = {f.name: [None] for f in NARRATORS_CANONICAL_SCHEMA}
    existing["canonical_id"] = ["nar:preexisting"]
    existing["source_ids"] = [["sanadset:9"]]
    existing["aliases"] = [[]]
    existing["mention_count"] = [5]
    pq.write_table(
        pa.table(existing, schema=NARRATORS_CANONICAL_SCHEMA),
        out_dir / "narrators_canonical.parquet",
    )

    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:1", "source": "itqan", "name_ar": "ج", "name_ar_normalized": "ج"}],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    ids = {r["canonical_id"] for r in pq.read_table(path).to_pylist()}
    assert "nar:preexisting" in ids  # not clobbered
    assert len(ids) == 2  # plus the new itqan-derived narrator


def test_no_bio_files_returns_none(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    assert promote_bios_to_canonical(staging, out_dir) is None


def test_aliases_attach_to_matching_canonical(tmp_path: Path) -> None:
    # da#94: name variants emitted by the Itqan parser land on the SAME canonical
    # narrator the bio produced (keyed by canonical_name_ar_normalized).
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "سعيد بن سماك"
    norm = normalize_arabic(name)
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": norm,
            }
        ],
    )
    _write_aliases(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "canonical_name_ar_normalized": norm,
                "alias": "ابن سماك",
                "alias_normalized": normalize_arabic("ابن سماك"),
            },
            # A variant whose normalized form equals the primary name is skipped.
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "canonical_name_ar_normalized": norm,
                "alias": name,
                "alias_normalized": norm,
            },
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rec = next(
        r for r in pq.read_table(path).to_pylist() if r["canonical_id"] == make_canonical_id(norm)
    )
    assert rec["aliases"] == [normalize_arabic("ابن سماك")]


def test_alias_without_matching_bio_is_dropped(tmp_path: Path) -> None:
    # An alias never *creates* a narrator — only a promoted bio anchors one.
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:1", "source": "itqan", "name_ar": "ج", "name_ar_normalized": "ج"}],
    )
    _write_aliases(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:999",
                "source": "itqan",
                "canonical_name_ar_normalized": "لا أحد",
                "alias": "x",
                "alias_normalized": "x",
            }
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    recs = pq.read_table(path).to_pylist()
    assert len(recs) == 1  # only the bio-anchored narrator
    assert recs[0]["aliases"] == []


def test_alias_source_filter_respected(tmp_path: Path) -> None:
    # With sources={"itqan"}, a non-itqan alias file must not be merged.
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "محمد"
    norm = normalize_arabic(name)
    _write_bios(
        staging,
        "itqan",
        [{"bio_id": "itqan:7", "source": "itqan", "name_ar": name, "name_ar_normalized": norm}],
    )
    _write_aliases(
        staging,
        "other",
        [
            {
                "bio_id": "other:7",
                "source": "other",
                "canonical_name_ar_normalized": norm,
                "alias": "أبو القاسم",
                "alias_normalized": normalize_arabic("أبو القاسم"),
            }
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir, sources={"itqan"})
    assert path is not None
    rec = next(
        r for r in pq.read_table(path).to_pylist() if r["canonical_id"] == make_canonical_id(norm)
    )
    assert rec["aliases"] == []


def test_unset_precision_defaults_to_unknown(tmp_path: Path) -> None:
    """da#239: bio-only narrators carry no precision key, so the builder must
    default both precision columns to UNKNOWN — never null (matches disambiguate
    and the da#161 Narrator non-Optional invariant)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    name = "سعيد بن سماك"
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": normalize_arabic(name),
            }
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rec = pq.read_table(path).to_pylist()[0]
    assert rec["birth_date_precision"] == DatePrecision.UNKNOWN.value
    assert rec["death_date_precision"] == DatePrecision.UNKNOWN.value
    assert rec["birth_date_precision"] is not None
    assert rec["death_date_precision"] is not None


def test_bio_promote_applies_name_quality_filter(tmp_path: Path) -> None:
    """da#271: bio_promote runs clean_narrator_name — pollution dropped, benediction
    suffix cleaned so the bio merges onto the clean canonical id (not a fork)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    clean_name = "انس بن مالك"
    _write_bios(
        staging,
        "kaggle",
        [
            # pure Prophet-reference "name" (kaggle rijāl pollution) → dropped
            {
                "bio_id": "kaggle:1",
                "source": "kaggle_narrators",
                "name_ar": "رسول الله صلى الله عليه واله",
                "name_ar_normalized": normalize_arabic("رسول الله صلى الله عليه واله"),
            },
            # a real name with a benediction suffix → cleaned to the bare name
            {
                "bio_id": "kaggle:2",
                "source": "kaggle_narrators",
                "name_ar": "انس بن مالك رضي الله عنه",
                "name_ar_normalized": normalize_arabic("انس بن مالك رضي الله عنه"),
            },
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rows = pq.read_table(path).to_pylist()

    ids = {r["canonical_id"] for r in rows}
    # the Prophet-reference bio produced no canonical node
    assert make_canonical_id(normalize_arabic("رسول الله صلى الله عليه واله")) not in ids
    # the benediction-suffixed bio promoted under the CLEAN name's id (merge-ready)
    assert make_canonical_id(clean_name) in ids
    names = {r["name_ar_normalized"] for r in rows}
    assert clean_name in names
