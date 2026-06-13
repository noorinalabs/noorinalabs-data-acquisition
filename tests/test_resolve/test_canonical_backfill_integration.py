"""Integration test for the canonical_narrator_id backfill (#109).

Runs the real ``ner -> disambiguate`` pipeline on fixtures and asserts that a
resolved mention ends up carrying its canonical ``nar:<uuid5>`` id, so the
downstream graph loaders can materialize a real ``(:Narrator)-[:NARRATED]->
(:Hadith)`` edge and a non-empty Chain.

Why this test is shaped this way:

- It does **not** hand-inject ``canonical_narrator_id`` into the mention rows.
  That hand-injection (present in the graph-loader unit fixtures over in
  noorinalabs-isnad-graph) is exactly the anti-pattern that masked #109 — it
  made the loaders look wired while the real pipeline produced ``None``. Here
  the canonical id is computed by the actual ``disambiguate.run`` and read back
  off the on-disk artifact.
- The graph edge/chain loaders themselves live in ``noorinalabs-isnad-graph``
  (``src/graph/load_edges.py::_load_narrated`` / ``load_nodes.py::_load_chains``),
  not in this repo, so a live Neo4j ``load`` cannot run here. Instead we replicate
  the documented loader *contract* — read ``canonical_narrator_id`` straight off
  the resolved-mention rows — and assert it now yields edges. Before the backfill
  this contract produces zero edges (the bug); after it, edges fire.

Remove the backfill in ``disambiguate.run`` and
``test_disambiguate_backfills_canonical_id_into_mentions`` fails (canonical ids
stay ``None`` -> zero edges).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.resolve import disambiguate, ner
from src.utils.arabic import normalize_arabic

# Two narrators in a single hadith's chain. The mention names are chosen to
# exact-match the bio candidates after Arabic normalization.
_NAME_A = "محمد بن اسماعيل"
_NAME_B = "عبد الله بن يوسف"
_HADITH_ID = "hdt:test:H1"


def _write_phase1_mentions(staging_dir: Path) -> None:
    """Write a Phase-1 sanadset mention table (the NER input for one hadith)."""
    table = pa.table(
        {
            "name_ar": pa.array([_NAME_A, _NAME_B], type=pa.string()),
            "name_en": pa.array([None, None], type=pa.string()),
            "name_ar_normalized": pa.array(
                [normalize_arabic(_NAME_A), normalize_arabic(_NAME_B)], type=pa.string()
            ),
            "source_hadith_id": pa.array([_HADITH_ID, _HADITH_ID], type=pa.string()),
            "position_in_chain": pa.array([0, 1], type=pa.int32()),
            "transmission_method": pa.array(["حدثنا", "عن"], type=pa.string()),
        }
    )
    pq.write_table(table, staging_dir / "narrator_mentions_sanadset.parquet")


def _write_bio_candidates(staging_dir: Path) -> None:
    """Write a bio candidate table the two mentions resolve against."""

    def _col(values: list[str | int | None], typ: pa.DataType) -> pa.Array:
        return pa.array(values, type=typ)

    table = pa.table(
        {
            "bio_id": _col(["bio:a", "bio:b"], pa.string()),
            "name_ar": _col([_NAME_A, _NAME_B], pa.string()),
            "name_en": _col([None, None], pa.string()),
            "name_ar_normalized": _col(
                [normalize_arabic(_NAME_A), normalize_arabic(_NAME_B)], pa.string()
            ),
            "kunya": _col([None, None], pa.string()),
            "nisba": _col([None, None], pa.string()),
            "birth_year_ah": _col([None, None], pa.int32()),
            "death_year_ah": _col([256, 197], pa.int32()),
            "birth_location": _col([None, None], pa.string()),
            "death_location": _col([None, None], pa.string()),
            "generation": _col([None, None], pa.string()),
            "gender": _col(["male", "male"], pa.string()),
            "trustworthiness": _col([None, None], pa.string()),
            "external_id": _col([None, None], pa.string()),
            "source": _col(["test", "test"], pa.string()),
        }
    )
    pq.write_table(table, staging_dir / "narrators_bio_test.parquet")


def _read_mentions(output_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(output_dir / "narrator_mentions_resolved.parquet")
    return cast(list[dict[str, object]], table.to_pylist())


def _derive_narrated_edges(mentions: list[dict[str, object]]) -> set[tuple[str, str]]:
    """Replicate the graph loader's NARRATED contract (#109).

    ``_load_narrated`` keys a NARRATED edge on the *first* narrator of each
    hadith's chain (lowest ``position_in_chain``) via its
    ``canonical_narrator_id``. A ``None`` id yields no edge — which is precisely
    the pre-backfill behavior this test guards against.
    """
    first_by_hadith: dict[str, dict[str, object]] = {}
    for m in mentions:
        hid = str(m["hadith_id"])
        pos = int(m["position_in_chain"])  # type: ignore[call-overload]
        cur = first_by_hadith.get(hid)
        if cur is None or pos < int(cur["position_in_chain"]):  # type: ignore[call-overload]
            first_by_hadith[hid] = m

    edges: set[tuple[str, str]] = set()
    for hid, m in first_by_hadith.items():
        cid = m.get("canonical_narrator_id")
        if cid:
            edges.add((str(cid), hid))
    return edges


def _derive_chain_narrator_ids(mentions: list[dict[str, object]]) -> dict[str, list[str]]:
    """Replicate ``_load_chains``: collect non-null canonical ids per hadith."""
    by_hadith: dict[str, list[str]] = {}
    for m in sorted(mentions, key=lambda r: int(r["position_in_chain"])):  # type: ignore[call-overload]
        cid = m.get("canonical_narrator_id")
        if cid:
            by_hadith.setdefault(str(m["hadith_id"]), []).append(str(cid))
    return by_hadith


def test_disambiguate_backfills_canonical_id_into_mentions(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _write_phase1_mentions(staging)
    _write_bio_candidates(staging)

    # --- NER only: canonical ids are still the None placeholder -> ZERO edges.
    ner.run(staging, output)
    pre = _read_mentions(output)
    assert len(pre) == 2
    assert all(m["canonical_narrator_id"] is None for m in pre)
    assert _derive_narrated_edges(pre) == set(), "NER alone must not wire NARRATED"

    # --- disambiguate: backfills canonical ids -> NARRATED edge fires.
    disambiguate.run(staging, output)
    post = _read_mentions(output)
    assert len(post) == 2

    # Every resolved mention now carries a nar:<uuid5> canonical id + confidence.
    resolved = [m for m in post if m["canonical_narrator_id"]]
    assert len(resolved) == 2, "both mentions should resolve to a canonical narrator"
    for m in resolved:
        assert str(m["canonical_narrator_id"]).startswith("nar:")
        assert m["confidence"] is not None and float(m["confidence"]) >= 0.70  # type: ignore[arg-type]

    # The backfilled ids match the canonical narrator nodes (same id space).
    canonical = pq.read_table(output / "narrators_canonical.parquet").to_pylist()
    canonical_ids = {str(r["canonical_id"]) for r in canonical}
    assert {str(m["canonical_narrator_id"]) for m in resolved} <= canonical_ids

    # A real NARRATED edge now materializes (count > 0) — the bug fixed.
    edges = _derive_narrated_edges(post)
    assert len(edges) == 1
    first = min(post, key=lambda r: int(r["position_in_chain"]))  # type: ignore[call-overload]
    assert (str(first["canonical_narrator_id"]), _HADITH_ID) in edges

    # The Chain for the hadith has non-empty narrator_ids (chain_length > 0).
    chains = _derive_chain_narrator_ids(post)
    assert chains[_HADITH_ID]
    assert len(chains[_HADITH_ID]) == 2


def test_backfill_is_idempotent_on_reload(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _write_phase1_mentions(staging)
    _write_bio_candidates(staging)

    ner.run(staging, output)
    disambiguate.run(staging, output)
    first_rows = _read_mentions(output)
    first = {str(m["mention_id"]): m["canonical_narrator_id"] for m in first_rows}

    # Re-running disambiguation must not duplicate rows or change ids.
    disambiguate.run(staging, output)
    second_rows = _read_mentions(output)
    second = {str(m["mention_id"]): m["canonical_narrator_id"] for m in second_rows}

    assert len(second_rows) == 2
    assert first == second
    assert _derive_narrated_edges(second_rows) == _derive_narrated_edges(first_rows)
