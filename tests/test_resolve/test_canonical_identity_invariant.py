"""Canonical-identity invariant + bio-corroboration gate (da#356).

The defect
----------
``disambiguate`` derived a mention's canonical identity **and display name** from
the *matched biographical candidate* rather than from the mention itself. A
mention ``عائشة`` fuzzy-matching the OCR-corrupt itqan bio ``عائذة`` produced a
node keyed ``uuid5(عاءذه)`` and displayed ``عاءذه``; the mention's own — correct —
spelling was demoted to ``aliases``. Measured on the pre-fix corpus: 14,316
chimeric canonical nodes holding 593,456 mentions, of which ``crossref``
(Stage 5) produced 19,900 nodes / 474,126 mentions and ``fuzzy`` (Stage 2)
produced 1,741 / 137,422.

The invariant these tests pin
-----------------------------
Canonical identity and display name are a function of the **mention's**
normalized form — or of a *chain-evidence-driven* refinement of it (the da#248
mononym split, whose registry re-key is legitimate and deliberate). They are
**never** a function of a bio candidate's name. A bio match may attach
``external_id``, ``birth_year_ah``, ``death_year_ah``, ``generation``,
``gender``, ``trustworthiness`` and ``source_ids``. It may not rename, and it
may not re-key.

Fixtures use the real corpus spellings and the production Parquet schemas. The
name pairs below were verified to trip the intended stage before being written
down (``عاءشه``/``عاءذه`` → ratio 80.00, lev 1 → fuzzy; ``ابو جعفر محمد بن علي``
``الباقر``/``الكوفي`` → ratio 85.19, lev 4 → crossref, since fuzzy requires
lev ≤ 2).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.identity import make_canonical_id
from src.resolve import disambiguate
from src.resolve.mononym_split import MONONYM_REGISTRY, _temporally_plausible
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA
from src.utils.arabic import normalize_arabic

# --- Real corpus spellings -------------------------------------------------
# The attested, correct mention spellings.
AISHA = "عائشة"  # ʿĀʾisha bint Abī Bakr — 35,418 mentions in the pre-fix corpus
IBN_ABBAS = "ابن عباس"  # 49,826 mentions
BAQIR = "أبو جعفر محمد بن علي الباقر"  # Imam al-Bāqir

# The OCR-corrupt / different-person bio spellings these fuzzy- or crossref-match.
AISHA_CORRUPT = "عائذة"  # itqan OCR corruption of ʿĀʾisha  -> fuzzy, score 0.80
IBN_ABBAS_CORRUPT = "ابن عبس"  # itqan OCR corruption        -> fuzzy, score 0.93
KUFI = "أبو جعفر محمد بن علي الكوفي"  # a DIFFERENT man       -> crossref, score 0.85

# da#248 control: a registered mononym whose split is chain-evidence-driven.
SUFYAN = "سفيان"
# Fuzzy-matches `سفيان` (ratio 0.889, lev 1) but is NOT a MONONYM_REGISTRY key — the
# decoy that exposes the registry lookup keying on the bio instead of the mention.
SUFYAN_DECOY = "سفين"


def _bio_table(rows: list[dict[str, object]]) -> pa.Table:
    """Build a narrators_bio_* table matching the production column set."""
    cols: dict[str, pa.Array] = {}
    spec: list[tuple[str, pa.DataType]] = [
        ("bio_id", pa.string()),
        ("name_ar", pa.string()),
        ("name_en", pa.string()),
        ("name_ar_normalized", pa.string()),
        ("kunya", pa.string()),
        ("nisba", pa.string()),
        ("birth_year_ah", pa.int32()),
        ("death_year_ah", pa.int32()),
        ("birth_location", pa.string()),
        ("death_location", pa.string()),
        ("generation", pa.string()),
        ("gender", pa.string()),
        ("trustworthiness", pa.string()),
        ("external_id", pa.string()),
        ("source", pa.string()),
    ]
    for name, typ in spec:
        cols[name] = pa.array([r.get(name) for r in rows], type=typ)
    return pa.table(cols)


def _write_mentions(output_dir: Path, rows: list[dict[str, object]]) -> None:
    cols: dict[str, pa.Array] = {}
    for field in NARRATOR_MENTIONS_RESOLVED_SCHEMA:
        cols[field.name] = pa.array([r.get(field.name) for r in rows], type=field.type)
    pq.write_table(
        pa.table(cols, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA),
        output_dir / "narrator_mentions_resolved.parquet",
    )


def _mention(mid: str, hadith: str, pos: int, name: str) -> dict[str, object]:
    return {
        "mention_id": mid,
        "hadith_id": hadith,
        "source_corpus": "sunnah",
        "position_in_chain": pos,
        "name_raw": name,
        "name_normalized": normalize_arabic(name),
        "canonical_narrator_id": None,
        "transmission_method": "حدثنا",
        "confidence": None,
    }


def _dirs(tmp_path: Path, name: str) -> tuple[Path, Path]:
    staging = tmp_path / name / "staging"
    output = tmp_path / name / "output"
    staging.mkdir(parents=True)
    output.mkdir(parents=True)
    return staging, output


def _canonical(output_dir: Path) -> dict[str, dict[str, object]]:
    rows = pq.read_table(output_dir / "narrators_canonical.parquet").to_pylist()
    return {r["canonical_id"]: r for r in rows}


def _mentions_out(output_dir: Path) -> list[dict[str, object]]:
    return pq.read_table(output_dir / "narrator_mentions_resolved.parquet").to_pylist()


# ---------------------------------------------------------------------------
# The identity invariant
# ---------------------------------------------------------------------------
class TestCanonicalIdentityInvariant:
    def test_fuzzy_bio_match_does_not_rekey_or_rename(self, tmp_path: Path) -> None:
        """A mention fuzzy-matching a corrupt bio keeps its OWN id and display name."""
        staging, output = _dirs(tmp_path, "fuzzy")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:1",
                        "name_ar": AISHA_CORRUPT,
                        "name_en": "Aidha",
                        "name_ar_normalized": normalize_arabic(AISHA_CORRUPT),
                        "death_year_ah": 58,
                        "gender": "female",
                        "external_id": "itqan-9001",
                        "source": "itqan",
                    }
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, AISHA)])

        disambiguate.run(staging, output, batch_size=8)

        mention_norm = normalize_arabic(AISHA)
        expected_id = make_canonical_id(mention_norm)
        canon = _canonical(output)

        assert expected_id in canon, (
            f"canonical node must be keyed on the MENTION's form {mention_norm!r}; "
            f"got nodes {[(c, r['name_ar_normalized']) for c, r in canon.items()]}"
        )
        node = canon[expected_id]
        assert node["name_ar_normalized"] == mention_norm
        assert node["name_ar"] == AISHA, "display name must be the mention's, not the bio's"
        assert make_canonical_id(normalize_arabic(AISHA_CORRUPT)) not in canon

        # The mention points at its own node.
        assert _mentions_out(output)[0]["canonical_narrator_id"] == expected_id

    def test_crossref_bio_match_does_not_rekey_or_rename(self, tmp_path: Path) -> None:
        """Stage 5 crossref — 92% of chimeric nodes pre-fix — must also not re-key.

        crossref requires an ``external_id`` on the candidate, has no Levenshtein
        bound, and pre-fix ran through neither the temporal nor the geographic
        filter. al-Bāqir and al-Kūfī are different men at ratio 85.19 / lev 4.
        """
        staging, output = _dirs(tmp_path, "crossref")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:2",
                        "name_ar": KUFI,
                        "name_en": "al-Kufi",
                        "name_ar_normalized": normalize_arabic(KUFI),
                        "death_year_ah": 280,
                        "external_id": "itqan-9002",
                        "source": "itqan",
                    }
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, BAQIR)])

        disambiguate.run(staging, output, batch_size=8)

        mention_norm = normalize_arabic(BAQIR)
        expected_id = make_canonical_id(mention_norm)
        canon = _canonical(output)
        assert expected_id in canon, "crossref must not re-key the mention onto the bio"
        assert canon[expected_id]["name_ar"] == BAQIR
        assert make_canonical_id(normalize_arabic(KUFI)) not in canon

    def test_no_canonical_node_is_chimeric(self, tmp_path: Path) -> None:
        """CHIMERA INVARIANT.

        Every canonical node's ``name_ar_normalized`` must appear among its own
        mentions' normalized surfaces — unless it is a da#248 registry-refined
        person, whose re-key is chain-evidence-driven rather than bio-driven.
        """
        staging, output = _dirs(tmp_path, "chimera")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:1",
                        "name_ar": AISHA_CORRUPT,
                        "name_en": "Aidha",
                        "name_ar_normalized": normalize_arabic(AISHA_CORRUPT),
                        "death_year_ah": 58,
                        "external_id": "itqan-9001",
                        "source": "itqan",
                    },
                    {
                        "bio_id": "bio:itqan:2",
                        "name_ar": IBN_ABBAS_CORRUPT,
                        "name_en": "Ibn Abs",
                        "name_ar_normalized": normalize_arabic(IBN_ABBAS_CORRUPT),
                        "death_year_ah": 68,
                        "external_id": "itqan-9002",
                        "source": "itqan",
                    },
                    {
                        "bio_id": "bio:itqan:3",
                        "name_ar": KUFI,
                        "name_en": "al-Kufi",
                        "name_ar_normalized": normalize_arabic(KUFI),
                        "death_year_ah": 280,
                        "external_id": "itqan-9003",
                        "source": "itqan",
                    },
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(
            output,
            [
                _mention("m:1", "hdt:sunnah:1", 0, AISHA),
                _mention("m:2", "hdt:sunnah:1", 1, IBN_ABBAS),
                _mention("m:3", "hdt:sunnah:2", 0, BAQIR),
                _mention("m:4", "hdt:sunnah:2", 1, AISHA),
            ],
        )

        disambiguate.run(staging, output, batch_size=2)

        canon = _canonical(output)
        surfaces: dict[str, set[str]] = {}
        for m in _mentions_out(output):
            cid = m["canonical_narrator_id"]
            assert cid is not None
            raw = m["name_normalized"] or m["name_raw"] or ""
            surfaces.setdefault(str(cid), set()).add(normalize_arabic(str(raw)))

        refined = {p.norm_name for persons in MONONYM_REGISTRY.values() for p in persons}
        chimeric = [
            (cid, row["name_ar_normalized"], sorted(surfaces.get(cid, set())))
            for cid, row in canon.items()
            if row["name_ar_normalized"] not in refined
            and row["name_ar_normalized"] not in surfaces.get(cid, set())
        ]
        assert not chimeric, f"chimeric canonical nodes (name absent from own mentions): {chimeric}"

    def test_display_name_never_taken_from_bio(self, tmp_path: Path) -> None:
        """``name_ar`` / ``name_en`` may come from the bio only on a name-identical match."""
        staging, output = _dirs(tmp_path, "display")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:1",
                        "name_ar": AISHA_CORRUPT,
                        "name_en": "Aidha",
                        "name_ar_normalized": normalize_arabic(AISHA_CORRUPT),
                        "death_year_ah": 58,
                        "external_id": "itqan-9001",
                        "source": "itqan",
                    }
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, AISHA)])
        disambiguate.run(staging, output, batch_size=8)

        node = _canonical(output)[make_canonical_id(normalize_arabic(AISHA))]
        assert node["name_ar"] == AISHA
        assert node["name_en"] != "Aidha", "a non-identical bio must not supply name_en"


# ---------------------------------------------------------------------------
# The corroboration gate
# ---------------------------------------------------------------------------
class TestCorroborationGate:
    def test_exact_match_still_attaches_bio_metadata(self, tmp_path: Path) -> None:
        """No regression: a name-identical bio still enriches the node."""
        staging, output = _dirs(tmp_path, "exact")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:m:1",
                        "name_ar": AISHA,
                        "name_en": "Aisha bint Abi Bakr",
                        "name_ar_normalized": normalize_arabic(AISHA),
                        "death_year_ah": 58,
                        "gender": "female",
                        "trustworthiness": "sahabi",
                        "external_id": "ms-0001",
                        "source": "muhaddithat",
                    }
                ]
            ),
            staging / "narrators_bio_muhaddithat.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, AISHA)])
        disambiguate.run(staging, output, batch_size=8)

        node = _canonical(output)[make_canonical_id(normalize_arabic(AISHA))]
        assert node["death_year_ah"] == 58
        assert node["gender"] == "female"
        assert node["external_id"] == "ms-0001"
        assert node["name_en"] == "Aisha bint Abi Bakr"
        assert node["source_ids"] == ["bio:m:1"]

    def test_weak_fuzzy_match_does_not_attach_bio_metadata(self, tmp_path: Path) -> None:
        """ʿĀʾisha↔ʿĀʾidha scores 0.80 — below the strict bar, no chain evidence.

        Identity is already safe; the gate additionally withholds the bio's dates
        so they cannot poison ``death_year_index`` (which feeds ``_temporal_filter``
        for chain neighbours *and* ``refine_mononym_name``'s da#248 evidence).
        """
        staging, output = _dirs(tmp_path, "weak")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:1",
                        "name_ar": AISHA_CORRUPT,
                        "name_en": "Aidha",
                        "name_ar_normalized": normalize_arabic(AISHA_CORRUPT),
                        "death_year_ah": 58,
                        "external_id": "itqan-9001",
                        "source": "itqan",
                    }
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, AISHA)])
        disambiguate.run(staging, output, batch_size=8)

        node = _canonical(output)[make_canonical_id(normalize_arabic(AISHA))]
        assert node["death_year_ah"] is None, "uncorroborated bio metadata must not attach"
        assert node["external_id"] is None

    def test_strict_near_exact_fuzzy_match_attaches_bio_metadata(self, tmp_path: Path) -> None:
        """Ibn ʿAbbās↔Ibn ʿAbs scores 0.933 at lev 1 — the corrupt bio IS his bio."""
        staging, output = _dirs(tmp_path, "strict")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:itqan:2",
                        "name_ar": IBN_ABBAS_CORRUPT,
                        "name_en": "Ibn Abbas",
                        "name_ar_normalized": normalize_arabic(IBN_ABBAS_CORRUPT),
                        "death_year_ah": 68,
                        "external_id": "itqan-9002",
                        "source": "itqan",
                    }
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(output, [_mention("m:1", "hdt:sunnah:1", 0, IBN_ABBAS)])
        disambiguate.run(staging, output, batch_size=8)

        node = _canonical(output)[make_canonical_id(normalize_arabic(IBN_ABBAS))]
        assert node["name_ar_normalized"] == normalize_arabic(IBN_ABBAS)
        assert node["death_year_ah"] == 68, "a near-exact bio still enriches"
        assert node["external_id"] == "itqan-9002"

    def test_temporal_contradiction_vetoes_metadata(self, tmp_path: Path) -> None:
        """A bio whose death year is implausible against chain neighbours is not attached.

        Neighbour at position 0 is exactly dated (exact bio, d. 68). The position-1
        mention near-exact-matches a bio dated d. 900 — a ~832y gap, far outside the
        15–80y teacher/student band — so its metadata is vetoed even though the name
        score clears the strict bar.
        """
        staging, output = _dirs(tmp_path, "veto")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:anchor",
                        "name_ar": IBN_ABBAS,
                        "name_en": "Ibn Abbas",
                        "name_ar_normalized": normalize_arabic(IBN_ABBAS),
                        "death_year_ah": 68,
                        "external_id": "ms-0002",
                        "source": "muhaddithat",
                    },
                    {
                        "bio_id": "bio:absurd",
                        "name_ar": AISHA_CORRUPT,
                        "name_en": "Aidha",
                        "name_ar_normalized": normalize_arabic(AISHA_CORRUPT),
                        "death_year_ah": 900,
                        "external_id": "itqan-9001",
                        "source": "itqan",
                    },
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(
            output,
            [
                _mention("m:1", "hdt:sunnah:1", 0, IBN_ABBAS),
                _mention("m:2", "hdt:sunnah:1", 1, AISHA),
            ],
        )
        disambiguate.run(staging, output, batch_size=8)

        canon = _canonical(output)
        aisha = canon[make_canonical_id(normalize_arabic(AISHA))]
        assert aisha["death_year_ah"] is None, "temporally contradicted bio must be vetoed"


# ---------------------------------------------------------------------------
# da#248 regression guard — the one legitimate re-key must survive
# ---------------------------------------------------------------------------
class TestMononymSplitPreserved:
    def test_registered_mononym_still_splits_on_chain_evidence(self, tmp_path: Path) -> None:
        """``سفيان`` beside a dated neighbour still refines to the selected person.

        This is the carve-out the chimera invariant must tolerate: the node's name
        (``سفيان الثوري``) legitimately does not appear among its mentions
        (``سفيان``), because the re-key is driven by chain evidence, not by a bio.
        """
        persons = MONONYM_REGISTRY[normalize_arabic(SUFYAN)]
        # da#248 abstains unless the evidence fits EXACTLY ONE registered person.
        # Pick a neighbour year that discriminates, rather than assuming one does:
        # e.g. 131 is plausible for al-Thawrī (d.161, gap 30) AND ibn ʿUyayna
        # (d.198, gap 67), so it would correctly abstain and prove nothing.
        candidates = [
            y
            for y in range(60, 280)
            if len([p for p in persons if _temporally_plausible(p, [y])]) == 1
        ]
        assert candidates, "fixture precondition: some year must select exactly one person"
        neighbour_year = candidates[0]
        (target,) = [p for p in persons if _temporally_plausible(p, [neighbour_year])]

        staging, output = _dirs(tmp_path, "mononym")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:neighbour",
                        "name_ar": IBN_ABBAS,
                        "name_en": "Ibn Abbas",
                        "name_ar_normalized": normalize_arabic(IBN_ABBAS),
                        "death_year_ah": neighbour_year,
                        "external_id": "ms-0002",
                        "source": "muhaddithat",
                    }
                ]
            ),
            staging / "narrators_bio_muhaddithat.parquet",
        )
        _write_mentions(
            output,
            [
                _mention("m:1", "hdt:sunnah:1", 0, IBN_ABBAS),
                _mention("m:2", "hdt:sunnah:1", 1, SUFYAN),
            ],
        )
        disambiguate.run(staging, output, batch_size=8)

        canon = _canonical(output)
        refined_id = make_canonical_id(target.norm_name)
        bare_id = make_canonical_id(normalize_arabic(SUFYAN))
        assert refined_id in canon, (
            f"da#248 split must survive: expected {target.norm_name!r}; "
            f"got {[r['name_ar_normalized'] for r in canon.values()]}"
        )
        assert bare_id not in canon

    def test_mononym_split_keys_on_the_mention_not_the_matched_bio(self, tmp_path: Path) -> None:
        """The l.1214-after-l.1203 ordering bug: same defect as the main one, one call deeper.

        Pre-fix, `refine_mononym_name` was handed `norm_name` *after* that variable had
        already been overwritten with the matched bio's name. So the `MONONYM_REGISTRY`
        lookup keyed on the **bio** spelling. A mention `سفيان` (a registry key) that
        fuzzy-matched a bio spelled `سفين` (ratio 0.889, lev 1 — *not* a registry key)
        therefore looked up `سفين`, missed, and the da#248 split was silently
        **suppressed** — the mention landed on a chimeric `سفين` node instead.

        This is the inverse failure of the guard above: there the split must survive,
        here a split that should fire was being lost. Both follow from keying on the
        mention.
        """
        persons = MONONYM_REGISTRY[normalize_arabic(SUFYAN)]
        years = [
            y
            for y in range(60, 280)
            if len([p for p in persons if _temporally_plausible(p, [y])]) == 1
        ]
        assert years, "fixture precondition: some year must select exactly one person"
        neighbour_year = years[0]
        (target,) = [p for p in persons if _temporally_plausible(p, [neighbour_year])]

        staging, output = _dirs(tmp_path, "ordering")
        pq.write_table(
            _bio_table(
                [
                    {
                        "bio_id": "bio:neighbour",
                        "name_ar": IBN_ABBAS,
                        "name_en": "Ibn Abbas",
                        "name_ar_normalized": normalize_arabic(IBN_ABBAS),
                        "death_year_ah": neighbour_year,
                        "external_id": "ms-0002",
                        "source": "muhaddithat",
                    },
                    {
                        # Fuzzy-matches `سفيان` but is NOT a registry key. Undated, so the
                        # temporal filter passes it through (soft constraint) and the
                        # pre-fix code substitutes its name before the registry lookup.
                        "bio_id": "bio:decoy",
                        "name_ar": SUFYAN_DECOY,
                        "name_en": None,
                        "name_ar_normalized": normalize_arabic(SUFYAN_DECOY),
                        "death_year_ah": None,
                        "source": "itqan",
                    },
                ]
            ),
            staging / "narrators_bio_itqan.parquet",
        )
        _write_mentions(
            output,
            [
                _mention("m:1", "hdt:sunnah:1", 0, IBN_ABBAS),
                _mention("m:2", "hdt:sunnah:1", 1, SUFYAN),
            ],
        )
        disambiguate.run(staging, output, batch_size=8)

        canon = _canonical(output)
        assert normalize_arabic(SUFYAN_DECOY) not in {
            r["name_ar_normalized"] for r in canon.values()
        }, "the decoy bio's spelling must never become a canonical name"
        assert make_canonical_id(target.norm_name) in canon, (
            f"registry lookup must key on the mention {normalize_arabic(SUFYAN)!r}, not on the "
            f"matched bio {normalize_arabic(SUFYAN_DECOY)!r}; "
            f"got {[r['name_ar_normalized'] for r in canon.values()]}"
        )


# ---------------------------------------------------------------------------
# The gate as a pure function — the unit the run() tests exercise end-to-end
# ---------------------------------------------------------------------------
class TestBioCorroboratedUnit:
    def _match(self, stage: str, score: float, death: int | None, norm: str) -> disambiguate.Match:
        return disambiguate.Match(
            candidate=disambiguate.Candidate(
                bio_id="b", name_ar=norm, name_ar_normalized=norm, death_year_ah=death
            ),
            stage=stage,
            score=score,
        )

    def test_exact_always_corroborated(self) -> None:
        m = self._match("exact", 1.0, 58, "x")
        assert disambiguate._bio_corroborated(m, "x", [])

    def test_agreeing_chain_evidence_corroborates_a_weak_score(self) -> None:
        m = self._match("fuzzy", 0.80, 58, normalize_arabic(AISHA_CORRUPT))
        assert disambiguate._bio_corroborated(m, normalize_arabic(AISHA), [100])

    def test_contradicting_chain_evidence_vetoes_a_strong_score(self) -> None:
        m = self._match("fuzzy", 0.99, 900, normalize_arabic(AISHA_CORRUPT))
        assert not disambiguate._bio_corroborated(m, normalize_arabic(AISHA), [68])

    def test_no_evidence_falls_back_to_strict_score_and_distance(self) -> None:
        weak = self._match("fuzzy", 0.80, None, normalize_arabic(AISHA_CORRUPT))
        assert not disambiguate._bio_corroborated(weak, normalize_arabic(AISHA), [])

        strong = self._match("fuzzy", 0.9333, None, normalize_arabic(IBN_ABBAS_CORRUPT))
        assert disambiguate._bio_corroborated(strong, normalize_arabic(IBN_ABBAS), [])

    def test_strict_score_still_requires_near_identity(self) -> None:
        """High score but lev > 1 (al-Aʿmash↔al-Aʿlam) is not corroboration."""
        m = self._match("fuzzy", 0.95, None, normalize_arabic("الأعلم"))
        assert not disambiguate._bio_corroborated(m, normalize_arabic("الأعمش"), [])


@pytest.mark.parametrize(
    ("mention", "bio"),
    [(AISHA, AISHA_CORRUPT), (IBN_ABBAS, IBN_ABBAS_CORRUPT), (BAQIR, KUFI)],
)
def test_canonical_id_is_uuid5_of_the_mention_form(mention: str, bio: str) -> None:
    """The identity contract, stated directly: id keys on the mention, never the bio."""
    assert make_canonical_id(normalize_arabic(mention)) != make_canonical_id(normalize_arabic(bio))
