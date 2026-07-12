"""Tests for the bio-direct canonical promoter (da#92a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.models.enums import DatePrecision
from src.parse.identity import make_canonical_id
from src.parse.name_quality import (
    asserted_name_tokens,
    clean_narrator_name,
    cleaner_removed_content,
)
from src.parse.schemas import NARRATOR_ALIAS_SCHEMA, NARRATOR_BIO_SCHEMA
from src.resolve import MissingInputError
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
    # da#370: a bio-promoted record (no isnad mention) is tagged biographical_only.
    assert rec["attestation"] == "biographical_only"


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


def test_no_bio_files_raises_missing_input(tmp_path: Path) -> None:
    """da#361: absent bio shards = an upstream defect, not an empty result.

    RED before da#361: this returned a success-shaped ``None`` indistinguishable
    from a genuine empty (see the present-but-empty case below).
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    with pytest.raises(MissingInputError) as excinfo:
        promote_bios_to_canonical(staging, out_dir)
    assert excinfo.value.stage == "bio_promote"
    assert "narrators_bio_*.parquet" in str(excinfo.value)


def test_present_but_empty_bio_shard_still_succeeds(tmp_path: Path) -> None:
    """da#361 distinction: a bio shard that EXISTS but is source-filtered to zero
    promotable rows is a genuine empty — it must NOT raise. The input is present, so
    the stage completes honestly (writing its canonical table), unlike the absent
    case above which raises."""
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    # A real (schema-valid) bio shard whose only row is filtered out by ``sources``.
    _write_bios(staging, "itqan", [{"bio_id": "b1", "name_ar": "محمد", "source": "itqan"}])
    # Restrict to a source the shard does not carry: input present, zero promoted —
    # an honest success (a written canonical path), never a raise.
    result = promote_bios_to_canonical(staging, out_dir, sources={"not_a_real_source"})
    assert result is not None


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

    `سعيد بن سماك بن حرب` survives the cleaner unchanged, so the guard must not fire.
    Without this the guard would be indistinguishable from "never merge a bio", which
    would silently disable da#247's merge-onto-clean-node behaviour for every
    well-formed name in the corpus.

    Verbatim from `data/raw/itqan/profiles_abandoned.json`, id 320 — NOT
    `profiles_companion.json`, which holds only 60592 and 58526. Its `grade_ar` is
    `متروك الحديث`, which is where this test's `matruk` comes from (Kavitha
    Sundaramurthy, da#379 review).
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


# Verbatim from data/staging/narrators_bio_itqan.parquet (bio_id itqan:25339). An
# itqan bracketed catalog entry number, NOT prose. `عبد الله بن موسى` is an attested
# narrator (2,712 mentions on the pre-PR#363 canonical table). da#379 review.
_ITQAN_25339_ENTRY_NUMBER = "( 4875 ) عبد الله بن موسى"
_ITQAN_25339_NAME = "عبد الله بن موسى"


def test_entry_number_prefix_is_not_a_truncation(tmp_path: Path) -> None:
    """da#379 review: a bracketed catalog entry number is an affix, not a tail.

    The cleaner strips `( 4875 )` and the residue is the *complete* name the source
    asserted. Refusing the merge here starves an attested narrator of his bio
    enrichment — the same harm the guard exists to prevent, with the sign flipped.
    """
    normalized = normalize_arabic(_ITQAN_25339_ENTRY_NUMBER)
    cleaned = clean_narrator_name(normalized)
    assert cleaned == _ITQAN_25339_NAME, f"fixture no longer strips the entry number: {cleaned!r}"
    assert not cleaner_removed_content(normalized, cleaned)

    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    attested_id = _write_attested_narrator(out_dir, _ITQAN_25339_NAME, mentions=2712)
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:25339",
                "source": "itqan",
                "name_ar": _ITQAN_25339_ENTRY_NUMBER,
                "name_ar_normalized": normalized,
                "trustworthiness": "daif",
                "external_id": "25339",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    # The merge proceeds and the back-fill lands: the bio enriches its own narrator.
    assert attested["source_ids"] == ["sanadset:1", "itqan:25339"]
    assert attested["trustworthiness"] == "daif"
    assert attested["external_id"] == "25339"


def test_honorific_suffix_is_not_a_truncation(tmp_path: Path) -> None:
    """A clean nasab carrying a benediction must still merge onto its attested node.

    No staging row exercises this today, which is exactly why it is pinned: the
    honorific strip is a normalization, and da#247 added the bio-side cleaner call
    *so that* "a benediction-suffixed bio merges onto the clean mention-derived node
    instead of forking it". A predicate that re-tokenizes the raw input sees the
    benediction as removed content and silently undoes that.
    """
    raw = "مالك بن انس رضي الله عنه"
    normalized = normalize_arabic(raw)
    cleaned = clean_narrator_name(normalized)
    assert cleaned == "مالك بن انس"
    assert not cleaner_removed_content(normalized, cleaned)

    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    attested_id = _write_attested_narrator(out_dir, "مالك بن انس", mentions=4000)
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:9001",
                "source": "itqan",
                "name_ar": raw,
                "name_ar_normalized": normalized,
                "trustworthiness": "thiqa",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    assert attested["trustworthiness"] == "thiqa"


def test_cleaner_removed_content_tracks_the_cleaners_own_tokenization() -> None:
    """The predicate must be computed against tokens the cleaner actually produces.

    This is the property, not the symptom: every transformation `clean_narrator_name`
    applies BEFORE it starts cutting (markup, honorific, editorial connective,
    leading ordinal) leaves the asserted name intact and must not read as removed
    content. A re-tokenization of the raw input fails every row below.
    """
    normalizations = [
        "( 4875 ) عبد الله بن موسى",  # bracketed entry number
        "[ 969 ] احمد بن يوسف الثعلبي",  # square-bracket entry number
        "مالك يعني بن انس",  # editorial connective
        "مالك بن انس رضي الله عنه",  # honorific suffix
    ]
    for raw in normalizations:
        normalized = normalize_arabic(raw)
        cleaned = clean_narrator_name(normalized)
        assert cleaned is not None, raw
        assert not cleaner_removed_content(normalized, cleaned), f"normalization read as cut: {raw}"
        assert tuple(cleaned.split()) == asserted_name_tokens(normalized)

    truncations = [
        "عبيده مولى رسول الله ذكره بن شاهين واستدركه ابو موسى",
        "سفيان يقال نفير بن مجيب الثمالي",
        "ابو هريره عن مكحول وعنه ابو المليح الرقي",
    ]
    for raw in truncations:
        normalized = normalize_arabic(raw)
        cleaned = clean_narrator_name(normalized)
        assert cleaned is not None, raw
        assert cleaner_removed_content(normalized, cleaned), f"truncation not detected: {raw}"


# Verbatim from data/staging/narrators_bio_itqan.parquet (bio_id itqan:12058). The
# residue `ابو نهشل` is NOT string-equal to the attested `ابي نهشل`, but da#376's
# inflection fold inside make_canonical_id gives them the same canonical id.
_ITQAN_12058_TRUNCATION = "أبو نهشل ، عن أبي وائل"


def test_truncated_residue_blocked_via_canonical_fold_not_string_equality(tmp_path: Path) -> None:
    """da#379 × da#376: the guard must key through `make_canonical_id`, not the raw string.

    `ابو نهشل ، عن ابي واءل` truncates at the isnad connective to `ابو نهشل`. The
    attested narrator is stored as `ابي نهشل` — a DIFFERENT string. Before da#376's
    fold these were two nodes and the guard had nothing to refuse; after it they are
    one, and the truncated residue would claim him. A guard that compared normalized
    names instead of canonical ids would silently pass this through.
    """
    from src.parse.identity import make_canonical_id as _mcid

    normalized = normalize_arabic(_ITQAN_12058_TRUNCATION)
    cleaned = clean_narrator_name(normalized)
    assert cleaned == "ابو نهشل"
    assert cleaner_removed_content(normalized, cleaned)
    # The discriminator: different strings, same identity.
    assert "ابو نهشل" != "ابي نهشل"
    assert _mcid("ابو نهشل") == _mcid("ابي نهشل"), "da#376 fold no longer collapses these"

    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    # The attested narrator is registered under the OTHER surface form.
    attested_id = _write_attested_narrator(out_dir, "ابي نهشل", mentions=8)
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:12058",
                "source": "itqan",
                "name_ar": _ITQAN_12058_TRUNCATION,
                "name_ar_normalized": normalized,
                "trustworthiness": "thiqa",
            }
        ],
    )
    rows = pq.read_table(promote_bios_to_canonical(staging, out_dir)).to_pylist()  # type: ignore[arg-type]
    attested = next(r for r in rows if r["canonical_id"] == attested_id)
    assert attested["source_ids"] == ["sanadset:1"], "truncated residue claimed him via the fold"
    assert attested["trustworthiness"] is None
    assert len(rows) == 1


# The DUAL of `test_cleaner_removed_content_tracks_the_cleaners_own_tokenization`.
#
# That test pins one direction: a NORMALIZATION must never read as removed content.
# This pins the other: a TRUNCATION must always read as removed content. Without it,
# `asserted_name_tokens` can quietly absorb one of the cleaner's four cuts — it exists
# to mirror the cleaner's prefix, so sharing *more* of that prefix is the obvious next
# tidy-up — and the refusals it was carrying evaporate against a green suite.
#
# Alejandra Reyes-Fuentes measured the exposure on the trailing-connective pop alone:
# 991 bio rows refused solely because of it, 42 of them targeting attested narrators
# (`ابو عمرو` at mc=899, `عبد الله بن محمد العدوي` at mc=1,194). Leak that cut and the
# whole tree stays green. A rule whose deletion no test notices is not under test.
#
# PROVENANCE — every row, with its id, and the scan's own scope stated below it.
#
# SCOPE OF THE SEARCH (stated because a zero whose search space is implicit cannot be
# audited — by a reviewer, or by its author an hour later; a provenance claim must carry
# its own provenance):
#   searched: the 32 parquet files under `data/staging/` and `data/curated/`
#   columns : name_ar, name_ar_normalized, name_raw, name_normalized, alias,
#             alias_normalized, canonical_name_ar_normalized, aliases
#   excluded: the 8 archival snapshot dirs (`data/curated.pre-da311-scrub-*`,
#             `data/curated.run5-scrubbed`, `data/curated.pre-wave23-reload-*`, …).
#             They are DELIBERATELY out of scope: a matn-derived name in a pre-scrub
#             snapshot is the defect da#311 removed, so a hit there is evidence of the
#             bug, not of the value being real. Widening to them manufactures false
#             positives that carry the authority of an exhaustive search.
#   control : read out of `narrator_aliases_itqan.parquet` — a file an earlier, narrower
#             scan never opened. A control drawn from inside the scope proves only that
#             the scan can see. It cannot prove the scope is right. The control must be a
#             value the scan would MISS if the scope were too narrow.
#
# This block has been wrong twice, in opposite directions, and both errors were one error:
# a claim about a SET, written while looking at one MEMBER.
#   1. It named the Thawban row "the one fixture not drawn from disk." Naming one
#      exception silently certifies every row you did not name.
#   2. Correcting that, it listed the apposition fixture as NOT on disk, from a scan that
#      never opened the alias table — and whose instrument guard passed, because the
#      control lived inside the subset being searched.
#
# ATTESTED ON DISK, AND ON THIS GUARD'S PATH — verbatim `name_ar` from
# `data/staging/narrators_bio_itqan.parquet`, which is the column `bio_promote` feeds the
# cleaner (`bio_promote.py:168`):
#   `أبو عمرو الذي`                  itqan:45083
#   `عبد الله بن محمد العدوي يقال`   itqan:47593
#   `أبو نصر الذي`                   itqan:47050
#   `ابو محمد بصري عن محمد بن علي`   itqan:49877
#   `مكاتب أم سلمة زوج النبي`        itqan:70033
#
#       The apposition row was `عاءشه زوج النبي` until this commit. That string IS on
#       disk — itqan:342, `narrator_aliases_itqan.parquet::alias_normalized`, an alias of
#       `عاءشه بنت ابي بكر الصديق`, and it reaches `narrators_canonical.aliases`. But it
#       is NOT on this guard's path: no `narrators_bio_*` name column carries it, and
#       aliases are a different pass. "On disk" alone would over-claim that the bio path
#       sees it; "not on disk" was simply false. `itqan:70033` is both, so the ambiguity
#       is retired rather than annotated. The cut is not hypothetical: it fires on 301
#       rows of the bio name path today.
#
# NOT ON DISK — one row, `Thawban:The Messenger…`, the colon-prose cut (da#253), quoted
# verbatim from `clean_narrator_name`'s own docstring.
#
#   State the zero precisely, because the obvious phrasing is false. It is NOT true that
#   no name field contains a colon: 6,814 rows do (5,383 itqan bio, 1,426 sanadset
#   mentions, 5 lk), and they are `، ويقال :` variant-name lists. The true claim is about
#   the CUT, not the character — `_truncate_colon_prose` requires a colon whose following
#   segment opens with an English non-name leader, and it fires on **0 of 140,174** bio
#   rows. The detector was checked against the fixture before that zero was believed.
#
#   A reviewer auditing "zero hits" by grepping for `:` finds 6,814 and concludes the
#   comment lied. A true conclusion resting on a false premise destroys the credibility
#   of the thing it was defending, and this block's whole job is to be believed.
#
#   Nor is the fixture merely a contract guard. `cleaner_removed_content`'s ONLY caller is
#   `bio_promote`, which passes `name_ar_normalized` verbatim (`bio_promote.py:168`), and
#   that field is not Arabic-only: 24,326 / 24,326 kaggle bio rows carry Latin script
#   (`Prophet Muhammad(saw) ( محمد …`). The colon-join is absent from the CORPUS, not
#   excluded by the caller's TYPE. One colon-joined kaggle row and this cut fires in
#   production, through the only call site there is.
#
# "No artifact DOES produce it" and "no artifact CAN produce it" are different sentences.
# Only the second makes an assertion inert (`feedback_fixture_makes_guard_assertion_inert`,
# fifth mode). The Thawban row is the first sentence, not the second.
@pytest.mark.parametrize(
    ("raw", "expected_cleaned", "cut"),
    [
        ("أبو عمرو الذي", "ابو عمرو", "trailing isnad connective (itqan:45083, target mc=899)"),
        (
            "عبد الله بن محمد العدوي يقال",
            "عبد الله بن محمد العدوي",
            "trailing isnad connective (itqan:47593, target mc=1,194)",
        ),
        ("أبو نصر الذي", "ابو نصر", "trailing isnad connective (itqan:47050, target mc=589)"),
        (
            "Thawban:The Messenger of Allah (ﷺ) sacrificed during a journey and then",
            "Thawban",
            "colon-joined English matn (da#253); NOT on disk — reachable, no instance",
        ),
        (
            "مكاتب أم سلمة زوج النبي",
            "مكاتب ام سلمه",
            "Prophet apposition (da#311); itqan:70033, on the bio name path",
        ),
        (
            "ابو محمد بصري عن محمد بن علي",
            "ابو محمد بصري",
            "isnad/matn boundary (da#258); itqan:49877",
        ),
    ],
)
def test_every_truncation_reads_as_removed_content(
    raw: str, expected_cleaned: str, cut: str
) -> None:
    """Each of the cleaner's four cuts must be visible to `cleaner_removed_content`.

    The residue of a truncation is a *head*, not the name the source asserted, so the
    guard must refuse to let it claim an attested narrator. `ابو عمرو` is a bare kunya
    — that is the da#379 defect exactly.
    """
    normalized = normalize_arabic(raw)
    cleaned = clean_narrator_name(normalized)
    assert cleaned == expected_cleaned, f"fixture no longer truncates ({cut}): {cleaned!r}"
    assert cleaner_removed_content(normalized, cleaned), f"truncation read as an affix: {cut}"
    # And the dual of the dual: the residue is strictly shorter than what was asserted.
    assert len(cleaned.split()) < len(asserted_name_tokens(normalized)), cut


# da#389 — verbatim from data/staging/narrators_bio_itqan.parquet +
# narrator_aliases_itqan.parquet, bio_id itqan:635. The bio name is a `، ويقال :`
# variant-list that `clean_narrator_name` truncates to its head nasab, so the node is
# minted under make_canonical_id(CLEANED) while the alias carried the RAW name — the
# exact shape of the 34,046 / 154,328 (22.1%) itqan aliases the pre-da#389 raw-key
# lookup silently dropped. A synthetic name that survived the cleaner unchanged would
# not exercise the re-key at all (raw-key == cleaned-key) and the test could not go red
# (feedback_fixture_makes_guard_assertion_inert).
_ITQAN_635_BIO_NAME = "هشام بن زياد بن أبي يزيد ، وهو هشام بن أبي هشام ، ويقال : هشام بن أبي الوليد"
_ITQAN_635_CLEANED = "هشام بن زياد بن ابي يزيد"
_ITQAN_635_ALIAS = "أبو المقدام"

# bio_id itqan:7532 — verbatim `name_ar`; a `، ويقال : / وقيل :` chain with no
# extractable head, so `clean_narrator_name` returns None (the bio is pollution-dropped
# and mints NO canonical node). Its alias therefore has nothing to anchor to — a
# LEGITIMATE non-attachment that must be *counted*, not silently lost.
_ITQAN_7532_POLLUTION = (
    "صرمة بن أنس ، ويقال : ابن أبي أنس ، ويقال :  ابن مالك ، ويقال : ابن قيس بن مالك بن "
    "عدي بن عامر بن غنم بن عدي بن النجار ،  وقيل : قيس بن صرمة ، وقيل : أبو قيس بن صرمة ، "
    "وقيل : أبو قيس بن عمرو"
)


def test_alias_rekeyed_to_cleaned_form_attaches(tmp_path: Path) -> None:
    """da#389: an alias whose bio-name the cleaner rewrote must still attach.

    RED on pre-da#389 code: the alias lookup keyed on make_canonical_id(RAW name), but
    the node was minted under make_canonical_id(clean_narrator_name(name)) — different
    ids — so the alias attached to nothing and was silently dropped.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    raw_norm = normalize_arabic(_ITQAN_635_BIO_NAME)
    # Premise (updated by da#371): the cleaner still rewrites this name, but da#371
    # centralized `clean_narrator_name` inside `make_canonical_id` (`canonical_key`), so
    # the raw and cleaned forms now mint the SAME id at the identity layer — da#389's
    # whole bug class (alias keyed on raw ≠ node minted on cleaned) is structurally gone,
    # and da#389's manual re-key to the cleaned form is now redundant-but-correct
    # (belt-and-suspenders). The alias-attach assertion below still proves the behavior.
    assert clean_narrator_name(raw_norm) == _ITQAN_635_CLEANED
    assert make_canonical_id(raw_norm) == make_canonical_id(_ITQAN_635_CLEANED)

    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:635",
                "source": "itqan",
                "name_ar": _ITQAN_635_BIO_NAME,
                "name_ar_normalized": raw_norm,
            }
        ],
    )
    _write_aliases(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:635",
                "source": "itqan",
                "canonical_name_ar_normalized": raw_norm,
                "alias": _ITQAN_635_ALIAS,
                "alias_normalized": normalize_arabic(_ITQAN_635_ALIAS),
            }
        ],
    )
    path = promote_bios_to_canonical(staging, out_dir)
    assert path is not None
    rec = next(
        r
        for r in pq.read_table(path).to_pylist()
        if r["canonical_id"] == make_canonical_id(_ITQAN_635_CLEANED)
    )
    assert rec["aliases"] == [normalize_arabic(_ITQAN_635_ALIAS)]


def test_merge_aliases_counts_recovered_and_pollution_dropped(tmp_path: Path) -> None:
    """da#389: both classes measured in one run — recovery AND the explained residual.

    The recovered alias attaches (``added == 1``); the alias whose bio the cleaner
    dropped is counted as ``unmatched_pollution`` (== 1) rather than vanishing; and no
    alias falls into the investigate bucket (``unmatched_no_bio == 0``). RED on
    pre-da#389 code: ``_AliasMergeStats`` does not exist and ``_merge_aliases`` returns
    a bare int with no drop accounting at all.
    """
    from src.resolve.bio_promote import _AliasMergeStats, _merge_aliases

    staging = tmp_path / "staging"
    staging.mkdir()

    pollution_norm = normalize_arabic(_ITQAN_7532_POLLUTION)
    assert clean_narrator_name(pollution_norm) is None  # bio mints no node

    _write_aliases(
        staging,
        "itqan",
        [
            # recovered: bio itqan:635 promoted under its CLEANED id → alias attaches
            {
                "bio_id": "itqan:635",
                "source": "itqan",
                "canonical_name_ar_normalized": normalize_arabic(_ITQAN_635_BIO_NAME),
                "alias": _ITQAN_635_ALIAS,
                "alias_normalized": normalize_arabic(_ITQAN_635_ALIAS),
            },
            # pollution: bio itqan:7532 was cleaner-dropped → counted, not silent
            {
                "bio_id": "itqan:7532",
                "source": "itqan",
                "canonical_name_ar_normalized": pollution_norm,
                "alias": "صرمة بن مالك",
                "alias_normalized": normalize_arabic("صرمة بن مالك"),
            },
        ],
    )

    cleaned_635 = clean_narrator_name(normalize_arabic(_ITQAN_635_BIO_NAME))
    assert cleaned_635 is not None
    cid_635 = make_canonical_id(cleaned_635)
    canonical_map: dict[str, dict[str, Any]] = {
        cid_635: {"canonical_id": cid_635, "name_ar_normalized": cleaned_635, "aliases": []}
    }

    stats = _merge_aliases(staging, canonical_map, {cid_635}, sources={"itqan"})
    assert isinstance(stats, _AliasMergeStats)
    assert stats.added == 1
    assert stats.unmatched_pollution == 1
    assert stats.unmatched_no_bio == 0
    assert canonical_map[cid_635]["aliases"] == [normalize_arabic(_ITQAN_635_ALIAS)]


def test_merge_aliases_drops_bare_relational_deixis(tmp_path: Path) -> None:
    """da#391: a bare relational-pronoun alias ("عمتي" = "my aunt") never attaches.

    The alias-list is the blind spot ``clean_narrator_name`` never reaches: the same
    unnamed-relation deixis it drops from a NAME would otherwise survive as an alias of
    a real narrator (on the ʿĀʾisha chimera: ``امتاه``/``خالته``/``عمتي``). It is dropped
    here, ACCOUNTED as ``skipped_deixis`` (not silent), and the sum-invariant holds.
    RED on pre-da#391 code: the variant is appended raw and ``skipped_deixis`` does not
    exist. A legitimate co-anchored variant on the SAME node still attaches, proving the
    scrub is surgical.
    """
    from src.resolve.bio_promote import _AliasMergeStats, _merge_aliases

    staging = tmp_path / "staging"
    staging.mkdir()

    real_variant = normalize_arabic(_ITQAN_635_ALIAS)
    anchor = normalize_arabic(_ITQAN_635_BIO_NAME)
    _write_aliases(
        staging,
        "itqan",
        [
            # bare relational deixis → dropped, counted as skipped_deixis
            {
                "bio_id": "itqan:635",
                "source": "itqan",
                "canonical_name_ar_normalized": anchor,
                "alias": "عمتي",
                "alias_normalized": normalize_arabic("عمتي"),
            },
            # a real co-anchored variant on the same node still attaches
            {
                "bio_id": "itqan:635",
                "source": "itqan",
                "canonical_name_ar_normalized": anchor,
                "alias": _ITQAN_635_ALIAS,
                "alias_normalized": real_variant,
            },
        ],
    )

    cleaned_635 = clean_narrator_name(anchor)
    assert cleaned_635 is not None
    cid_635 = make_canonical_id(cleaned_635)
    canonical_map: dict[str, dict[str, Any]] = {
        cid_635: {"canonical_id": cid_635, "name_ar_normalized": cleaned_635, "aliases": []}
    }

    stats = _merge_aliases(staging, canonical_map, {cid_635}, sources={"itqan"})
    assert isinstance(stats, _AliasMergeStats)
    assert stats.skipped_deixis == 1
    assert stats.added == 1
    # the deixis token is gone; only the real variant remains on the node
    assert canonical_map[cid_635]["aliases"] == [real_variant]
    # sum-invariant: every one of the 2 alias rows lands in exactly one bucket
    total = (
        stats.added
        + stats.skipped_duplicate
        + stats.skipped_empty
        + stats.skipped_source
        + stats.skipped_deixis
        + stats.unmatched_pollution
        + stats.unmatched_no_bio
    )
    assert total == 2


def test_unmatched_no_bio_is_counted_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """da#389: the `unmatched_no_bio` bucket AND the WARN it gates must be witnessed.

    The "never silent again" guarantee rests on this bucket and its warning, but every
    other test only asserts it is ZERO. An unwitnessed counter/branch can be deleted and
    leave the suite green — the silent-zero failure mode this program exists to kill
    (`feedback_silent_zero_is_not_a_measurement`). So drive it POSITIVE: an alias whose
    cleaned canonical name is a valid narrator name (NOT pollution) that simply no bio
    promoted — distinct from the `unmatched_pollution` case — and assert both the exact
    count and that the WARN actually fires.

    `مالك بن انس` and `سعيد بن جبير` are real attested narrators (used as fixtures across
    this module) and survive `clean_narrator_name` unchanged, so the non-attach here is
    "no promoted bio", never "cleaner-dropped".
    """
    from src.resolve import bio_promote as bp

    staging = tmp_path / "staging"
    staging.mkdir()
    out_dir = tmp_path / "curated"
    out_dir.mkdir()

    promoted_name = "سعيد بن جبير"
    orphan_alias_name = "مالك بن انس"  # cleans to itself; no bio promotes it below
    assert clean_narrator_name(normalize_arabic(orphan_alias_name)) == orphan_alias_name

    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "name_ar": promoted_name,
                "name_ar_normalized": normalize_arabic(promoted_name),
            }
        ],
    )
    _write_aliases(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "canonical_name_ar_normalized": normalize_arabic(orphan_alias_name),
                "alias": "أبو عبد الله",
                "alias_normalized": normalize_arabic("أبو عبد الله"),
            }
        ],
    )

    # Direct-call witness of the exact buckets: the orphan alias lands in
    # unmatched_no_bio, NOT unmatched_pollution, and nothing attaches.
    promoted_cid = make_canonical_id(normalize_arabic(promoted_name))
    stats = bp._merge_aliases(staging, {}, {promoted_cid}, sources={"itqan"})
    assert stats.unmatched_no_bio == 1
    assert stats.unmatched_pollution == 0
    assert stats.added == 0

    # End-to-end witness that the WARN branch actually fires (with the count). The
    # module logger is a structlog PrintLogger; spy on `.warning` directly so the
    # assertion does not depend on stdout-capture cache semantics.
    warnings_seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        bp.logger,
        "warning",
        lambda event, **kw: warnings_seen.append((event, kw)),
    )
    assert bp.promote_bios_to_canonical(staging, out_dir) is not None
    assert any(
        event == "bio_promote_aliases_unmatched_no_bio" and kw.get("unmatched_no_bio") == 1
        for event, kw in warnings_seen
    ), warnings_seen
