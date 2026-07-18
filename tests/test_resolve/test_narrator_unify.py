"""Tests for src.resolve.narrator_unify — the da#431/da#347 curated under-merge fix.

The load-bearing test is the **bidirectional acceptance gate**
(:class:`TestBidirectionalAcceptance`): a golden group of surface forms that are ONE
person MUST merge onto a single survivor bearing the summed mentions and the
biography, and every distinct-narrator control MUST NOT be swept in. The exclude
direction is sacred — merging two distinct narrators is the da#423 failure that
deleted the Prophet's daughter/son/Abū Bakr, each time with green CI. This mechanism
supplies curated external evidence to bypass fuzzy_cluster's precision guard ONLY for
the golden set; the gate (over-merge exclusion, corroborated-death veto, gross-spread
sanity band, gender veto, in-corpus evidence) refuses any group that fails, so a
curation error cannot silently over-merge.

Synthetic-only: no real corpus. The Arabic surfaces mirror the real Abū Wāʾil cluster
(da#431) so the golden case exercises the true kunya↔ism↔bio topology, but every row
is fabricated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import narrator_unify as nu
from src.resolve.schemas import (
    NARRATOR_MENTIONS_RESOLVED_SCHEMA,
    NARRATORS_CANONICAL_SCHEMA,
)

# --- Synthetic canonical rows (the real Abū Wāʾil cluster shape, fabricated) ----------

_KUNYA = "ابو واءل"
_ISM = "شقيق بن سلمه"
_BRIDGE = "ابو واءل شقيق بن سلمه"  # in-corpus full name carrying BOTH kunya and ism
_BIO = "شقيق ابو واءل بن سلمه الاسدي"  # rijāl bio node, 0 mentions
_BARE_SHAQIQ = "شقيق"  # a DIFFERENT Shaqīq (d.200) — must never merge in


def _canon(
    cid: str,
    norm: str,
    *,
    mc: int,
    death: int | None = None,
    prov: str | None = None,
    gender: str | None = None,
    corpus: str = "itqan",
) -> dict[str, Any]:
    row: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    row.update(
        canonical_id=cid,
        name_ar=norm,
        name_ar_normalized=norm,
        aliases=[],
        death_year_ah=death,
        death_year_provenance=prov,
        gender=gender,
        mention_count=mc,
        source_ids=[f"{corpus}:x:{cid}"],
        source_corpus=corpus,
        source_corpora=[corpus],
    )
    return row


def _write_canonical(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    arrays = {f.name: [r.get(f.name) for r in rows] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(
        pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA),
        output_dir / "narrators_canonical.parquet",
    )


def _write_mentions(output_dir: Path, pairs: list[tuple[str, str]]) -> None:
    """pairs = (mention_id, canonical_narrator_id)."""
    n = len(pairs)
    tbl = pa.table(
        {
            "mention_id": [m for m, _ in pairs],
            "hadith_id": [f"hdt:x:{i}" for i in range(n)],
            "source_corpus": ["itqan"] * n,
            "position_in_chain": [0] * n,
            "chain_index": [0] * n,
            "name_raw": [None] * n,
            "name_normalized": [None] * n,
            "canonical_narrator_id": [c for _, c in pairs],
            "transmission_method": [None] * n,
            "confidence": [None] * n,
        },
        schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA,
    )
    pq.write_table(tbl, output_dir / "narrator_mentions_resolved.parquet")


def _abu_wail_rows() -> list[dict[str, Any]]:
    return [
        _canon("nar:kunya", _KUNYA, mc=3117),  # holds the mentions, no bio
        _canon("nar:ism", _ISM, mc=364, death=82, gender="male"),  # ism+nasab
        _canon("nar:bridge", _BRIDGE, mc=58),  # bridging full name
        _canon("nar:bio", _BIO, mc=0, death=81),  # rijāl bio, 0 mentions
    ]


def _seed(tmp_path: Path, *, members: list[str]) -> Path:
    lines = [
        "unify:",
        '  - primary_label: "Abū Wāʾil (test)"',
        '    issue: "da#431"',
        "    evidence: bio_kunya_ism_bridge",
        f'    kunya_norm: "{_KUNYA}"',
        f'    ism_norm: "{_ISM}"',
        '    note: "synthetic test group"',
        "    members:",
        *[f'      - "{m}"' for m in members],
    ]
    path = tmp_path / "seed.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Golden: the merge MUST happen -----------------------------------------------------


class TestGoldenMerge:
    def test_kunya_ism_bio_merge_into_one(self, tmp_path: Path) -> None:
        out = tmp_path
        _write_canonical(out, _abu_wail_rows())
        _write_mentions(
            out,
            [("m1", "nar:kunya"), ("m2", "nar:kunya"), ("m3", "nar:ism"), ("m4", "nar:bridge")],
        )
        seed = _seed(tmp_path, members=[_KUNYA, _ISM, _BRIDGE, _BIO])

        result = nu.apply_narrator_unification(out, seed_path=seed)
        assert result is not None

        rows = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
        # 4 nodes collapse to 1 survivor.
        assert len(rows) == 1
        survivor = rows[0]
        # Mentions summed (3117 + 364 + 58 + 0).
        assert survivor["mention_count"] == 3117 + 364 + 58 + 0
        # Biography preserved from the bio/ism members despite the kunya being the rep.
        assert survivor["death_year_ah"] in (81, 82)
        # Every absorbed surface survives as an alias — no spelling lost.
        aliases = set(survivor["aliases"] or [])
        assert {_ISM, _BRIDGE, _BIO} <= aliases
        # attestation lifts off "biographical_only" (summed mc > 0).
        assert survivor["attestation"] == "isnad_attested"

    def test_mentions_remapped_to_survivor(self, tmp_path: Path) -> None:
        out = tmp_path
        _write_canonical(out, _abu_wail_rows())
        _write_mentions(out, [("m1", "nar:kunya"), ("m3", "nar:ism"), ("m4", "nar:bridge")])
        seed = _seed(tmp_path, members=[_KUNYA, _ISM, _BRIDGE, _BIO])

        nu.apply_narrator_unification(out, seed_path=seed)
        survivor_id = pq.read_table(out / "narrators_canonical.parquet").to_pylist()[0][
            "canonical_id"
        ]
        mention_ids = set(
            pq.read_table(out / "narrator_mentions_resolved.parquet")
            .column("canonical_narrator_id")
            .to_pylist()
        )
        # All mentions now point at the single survivor; no absorbed id remains.
        assert mention_ids == {survivor_id}

    def test_idempotent_second_run_is_noop(self, tmp_path: Path) -> None:
        out = tmp_path
        _write_canonical(out, _abu_wail_rows())
        _write_mentions(out, [("m1", "nar:kunya"), ("m3", "nar:ism")])
        seed = _seed(tmp_path, members=[_KUNYA, _ISM, _BRIDGE, _BIO])

        assert nu.apply_narrator_unification(out, seed_path=seed) is not None
        first = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
        # Second run: only the survivor matches a member surface -> < 2 nodes -> no-op.
        assert nu.apply_narrator_unification(out, seed_path=seed) is None
        second = pq.read_table(out / "narrators_canonical.parquet").to_pylist()
        assert first == second


# --- Bidirectional: the merge MUST NOT happen for a distinct narrator ------------------


class TestMustNotMerge:
    def _run(self, tmp_path: Path, rows: list[dict[str, Any]], members: list[str]) -> Path | None:
        out = tmp_path
        _write_canonical(out, rows)
        _write_mentions(out, [(f"m{i}", r["canonical_id"]) for i, r in enumerate(rows)])
        seed = _seed(tmp_path, members=members)
        return nu.apply_narrator_unification(out, seed_path=seed)

    def test_gross_death_spread_refuses_distinct_namesake(self, tmp_path: Path) -> None:
        # Bare Shaqīq d.200 (a different, taba-tabiʿī person) shares no gross-spread
        # tolerance with the d.82 Abū Wāʾil cluster: the group is refused wholesale.
        rows = _abu_wail_rows() + [_canon("nar:bare", _BARE_SHAQIQ, mc=1046, death=200)]
        assert self._run(tmp_path, rows, [_KUNYA, _ISM, _BRIDGE, _BIO, _BARE_SHAQIQ]) is None
        # nothing merged: all 5 nodes intact.
        assert len(pq.read_table(tmp_path / "narrators_canonical.parquet").to_pylist()) == 5

    def test_corroborated_death_conflict_refuses(self, tmp_path: Path) -> None:
        # Two members with CORROBORATED, conflicting death years within the sanity band
        # (82 vs 90, both corroborated) — still refused: a corroborated conflict is hard.
        rows = _abu_wail_rows()
        rows[1] = _canon("nar:ism", _ISM, mc=364, death=82, prov="corroborated", gender="male")
        rows.append(_canon("nar:bridge2", _BRIDGE, mc=10, death=90, prov="corroborated"))
        assert self._run(tmp_path, rows, [_KUNYA, _ISM, _BRIDGE, _BIO]) is None

    def test_over_merged_bare_generic_member_refuses(self, tmp_path: Path) -> None:
        # A member that is a curated over-merged bare generic (real seed: عبد الله) can
        # never be a unify member — the whole group is refused.
        rows = _abu_wail_rows() + [_canon("nar:generic", "عبد الله", mc=17752)]
        assert self._run(tmp_path, rows, [_KUNYA, _ISM, _BRIDGE, _BIO, "عبد الله"]) is None

    def test_gender_conflict_refuses(self, tmp_path: Path) -> None:
        rows = _abu_wail_rows()
        rows[0] = _canon("nar:kunya", _KUNYA, mc=3117, gender="male")
        rows.append(_canon("nar:fem", _BRIDGE, mc=5, gender="female"))
        assert self._run(tmp_path, rows, [_KUNYA, _ISM, _BRIDGE, _BIO]) is None

    def test_missing_bridge_is_uncorroborated_and_refuses(self, tmp_path: Path) -> None:
        # No member spells out BOTH kunya and ism (drop the bridge + bio) -> the
        # kunya↔ism identity is not attested in-corpus -> refused.
        rows = [
            _canon("nar:kunya", _KUNYA, mc=3117),
            _canon("nar:ism", _ISM, mc=364, death=82),
        ]
        assert self._run(tmp_path, rows, [_KUNYA, _ISM]) is None


# --- Seed parsing --------------------------------------------------------------------


class TestSeedParsing:
    def test_fewer_than_two_members_rejected(self, tmp_path: Path) -> None:
        seed = _seed(tmp_path, members=[_KUNYA])
        with pytest.raises(ValueError, match="2 members"):
            nu.load_unify_seed(seed)

    def test_bridge_evidence_requires_kunya_and_ism(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            'unify:\n  - primary_label: "x"\n    evidence: bio_kunya_ism_bridge\n'
            '    note: "n"\n    members: ["a", "b"]\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="kunya_norm and ism_norm"):
            nu.load_unify_seed(path)

    def test_missing_seed_file_is_empty(self, tmp_path: Path) -> None:
        assert nu.load_unify_seed(tmp_path / "nope.yaml") == ()
