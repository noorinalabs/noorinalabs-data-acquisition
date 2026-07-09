"""Tests for the bio-direct canonical promoter (da#92a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.parse.name_quality import clean_narrator_name
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
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "name_ar": "مالك بن انس",
                "name_ar_normalized": "مالك بن انس",
            }
        ],
    )
    _write_bios(
        staging,
        "muhaddithat",
        [
            {
                "bio_id": "m:1",
                "source": "muhaddithat",
                "name_ar": "حفصه بنت عمر",
                "name_ar_normalized": "حفصه بنت عمر",
            }
        ],
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
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "name_ar": "سعيد بن جبير",
                "name_ar_normalized": "سعيد بن جبير",
            }
        ],
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
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "name_ar": "سعيد بن جبير",
                "name_ar_normalized": "سعيد بن جبير",
            }
        ],
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


# Verbatim `full_name` values from the acquired Itqan buckets — NOT synthetic. A
# fabricated prose row would not exercise the truncation, and the guard could not go
# red (feedback_fixture_makes_guard_assertion_inert). Provenance:
#   itqan:60592 — data/raw/itqan/profiles_companion.json
#   itqan:58526 — data/raw/itqan/profiles_companion.json
_ITQAN_60592_PROSE = "عبيدة مولى رسول الله ذكره بن شاهين واستدركه أبو موسى وانما هو"
_ITQAN_58526_PROSE = "سفيان يقال نفير بن مجيب الثمالي"


def _write_attested_narrator(out_dir: Path, normalized_name: str, mentions: int) -> str:
    """Write a pre-existing mention-driven canonical record, as the disambiguator would."""
    from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA

    cid = make_canonical_id(normalized_name)
    existing = {f.name: [None] for f in NARRATORS_CANONICAL_SCHEMA}
    existing["canonical_id"] = [cid]
    existing["name_ar"] = [normalized_name]
    existing["name_ar_normalized"] = [normalized_name]
    existing["source_ids"] = [["sanadset:1"]]
    existing["aliases"] = [[]]
    existing["source_corpora"] = [[]]
    existing["mention_count"] = [mentions]
    pq.write_table(
        pa.table(existing, schema=NARRATORS_CANONICAL_SCHEMA),
        out_dir / "narrators_canonical.parquet",
    )
    return cid


def test_truncated_prose_bio_does_not_claim_attested_narrator(tmp_path: Path) -> None:
    """da#379: a prose bio truncating to a bare ism must not merge onto an attested narrator.

    `clean_narrator_name` cuts `عبيدة مولى رسول الله ذكره بن شاهين …` at the narrative
    boundary and keeps `عبيده` — the name of a *different*, heavily-attested narrator.
    Pre-guard, bio_promote keyed the prose row to that id and back-filled the obscure
    client's grade and death year onto him.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    truncated = clean_narrator_name(normalize_arabic(_ITQAN_60592_PROSE))
    assert truncated == "عبيده", f"fixture no longer truncates to a bare ism: {truncated!r}"

    attested_id = _write_attested_narrator(out_dir, "عبيده", mentions=1695)

    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:60592",
                "source": "itqan",
                "name_ar": _ITQAN_60592_PROSE,
                "name_ar_normalized": normalize_arabic(_ITQAN_60592_PROSE),
                "trustworthiness": "thiqa",
                "death_year_ah": 72,
                "generation": "sahabi",
                "external_id": "60592",
            }
        ],
    )

    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rows = pq.read_table(path).to_pylist()

    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    # The identity claim is refused: the bio never joins the attested narrator.
    assert attested["source_ids"] == ["sanadset:1"], "prose bio claimed an attested narrator"
    # And — the damage — the BACKFILL never fires. Asserting only on node creation
    # would pass over a fix that still corrupts the attested narrator's scholarship.
    assert attested["trustworthiness"] is None
    assert attested["death_year_ah"] is None
    assert attested["generation"] is None
    assert attested["external_id"] is None
    # The row is dropped, not re-keyed onto some other node.
    assert len(rows) == 1
    assert attested["mention_count"] == 1695


def test_truncated_prose_bio_dropped_for_second_real_row(tmp_path: Path) -> None:
    """Second verbatim row, different truncation path (`يقال` variant prose)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    assert clean_narrator_name(normalize_arabic(_ITQAN_58526_PROSE)) == "سفيان"
    attested_id = _write_attested_narrator(out_dir, "سفيان", mentions=28323)

    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:58526",
                "source": "itqan",
                "name_ar": _ITQAN_58526_PROSE,
                "name_ar_normalized": normalize_arabic(_ITQAN_58526_PROSE),
                "trustworthiness": "daif",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    assert attested["trustworthiness"] is None
    assert attested["source_ids"] == ["sanadset:1"]


def test_clean_bio_still_merges_onto_attested_narrator(tmp_path: Path) -> None:
    """The negative class: an untouched name still merges and back-fills as before.

    `سعيد بن سماك بن حرب` (itqan:320, verbatim) survives the cleaner unchanged, so the
    guard must not fire. Without this the guard would be indistinguishable from
    "never merge a bio", which would silently disable da#247's merge-onto-clean-node
    behaviour for every well-formed name in the corpus.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    name = "سعيد بن سماك بن حرب"
    normalized = normalize_arabic(name)
    assert clean_narrator_name(normalized) == normalized  # cleaner is a no-op here
    attested_id = _write_attested_narrator(out_dir, normalized, mentions=15)

    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:320",
                "source": "itqan",
                "name_ar": name,
                "name_ar_normalized": normalized,
                "trustworthiness": "matruk",
                "external_id": "320",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    assert attested["source_ids"] == ["sanadset:1", "itqan:320"]
    assert attested["trustworthiness"] == "matruk"  # backfill still works
    assert attested["external_id"] == "320"


def test_truncated_prose_bio_still_mints_when_target_unattested(tmp_path: Path) -> None:
    """Guard is scoped to attested targets — a zero-mention collision is out of scope.

    Widening it to zero-mention targets drops 12,043 kaggle rows for zero
    mention-bearing fixes; that corpus belongs to da#299.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    _write_attested_narrator(out_dir, "عبيده", mentions=0)
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:60592",
                "source": "itqan",
                "name_ar": _ITQAN_60592_PROSE,
                "name_ar_normalized": normalize_arabic(_ITQAN_60592_PROSE),
                "trustworthiness": "thiqa",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    merged = next(r for r in rows if r["canonical_id"] == make_canonical_id("عبيده"))
    assert merged["trustworthiness"] == "thiqa"  # unchanged behaviour
