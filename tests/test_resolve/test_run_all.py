"""Orchestrator tests for ``src.resolve.run_all`` wiring (da#117).

Covers the two invariants the wiring must hold:

1. ``run_all`` invokes both the bio promoter and the deterministic parallel
   detector, and runs ``disambiguate`` *before* ``bio_promote``.
2. The ordering is *safe*: a promoted bio-only narrator (one with no mention,
   so absent from the ``disambiguate`` output that OVERWRITES
   ``narrators_canonical.parquet``) survives a full ``run_all`` instead of being
   clobbered — proving ``bio_promote`` MERGEs *after* ``disambiguate`` (da#99).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.identity import make_canonical_id
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve import bio_promote, dedup, disambiguate, narrator_split, ner, parallels, run_all
from src.utils.arabic import normalize_arabic
from tests.test_graph.conftest import write_hadiths, write_narrator_mentions_resolved


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


def test_run_all_invokes_bio_promote_and_parallels_disambiguate_first(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """run_all wires in bio_promote + detect_parallels, disambiguate-first."""
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"
    # A staging Parquet so the pre-flight passes.
    write_hadiths(staging, [{"source_id": "h1", "matn_en": "x"}], suffix="t")

    calls: list[str] = []

    def _rec(name: str, ret: Any) -> Any:
        def _fn(*_a: Any, **_k: Any) -> Any:
            calls.append(name)
            return ret

        return _fn

    monkeypatch.setattr(ner, "run", _rec("ner", [staging / "mentions"]))
    monkeypatch.setattr(disambiguate, "run", _rec("disambiguate", []))
    monkeypatch.setattr(bio_promote, "promote_bios_to_canonical", _rec("bio_promote", None))
    # narrator_split now fails loud on an absent canonical table (da#361); this
    # disambiguate spy writes no canonical, so spy it too or run_all raises.
    monkeypatch.setattr(narrator_split, "split_generic_narrators", _rec("narrator_split", None))
    monkeypatch.setattr(dedup, "run", _rec("dedup", []))
    monkeypatch.setattr(parallels, "run", _rec("parallels", []))

    results = run_all(tmp_path / "raw", staging, curated)

    # Both new stages were invoked.
    assert "bio_promote" in calls
    assert "parallels" in calls
    # Ordering invariant: NER -> disambiguate -> bio_promote.
    assert calls.index("ner") < calls.index("disambiguate")
    assert calls.index("disambiguate") < calls.index("bio_promote")
    # The results dict exposes the new stages.
    assert {"bio_promote", "parallels"} <= results.keys()


def test_run_all_promoted_bio_only_narrator_survives(tmp_path: Path, monkeypatch: Any) -> None:
    """A bio-only narrator survives run_all — disambiguate runs before bio_promote.

    ``disambiguate`` OVERWRITES ``narrators_canonical.parquet`` with only the
    mentioned narrators; ``bio_promote`` MERGEs the bios on top. If the order
    were reversed, the overwrite would drop the bio-only narrator. Asserting it
    is present (alongside a disambiguate-produced narrator) proves the order.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    curated = tmp_path / "curated"

    mentioned_name = "زيد بن ثابت"  # has a mention -> emitted by disambiguate
    bio_only_name = "سعيد بن سماك"  # bio only, no mention -> only via bio_promote
    bio_only_id = make_canonical_id(normalize_arabic(bio_only_name))

    # Two cross-sect hadiths with identical matn -> a deterministic cross-sect
    # PARALLEL_OF edge, proving detect_parallels is in the pipeline path.
    write_hadiths(
        staging,
        [
            {"source_id": "hs1", "matn_en": "actions are by intentions", "sect": "sunni"},
            {"source_id": "hs2", "matn_en": "actions are by intentions", "sect": "shia"},
        ],
        suffix="x",
    )
    # Bio for the bio-only narrator (no corresponding mention).
    _write_bios(
        staging,
        "itqan",
        [
            {
                "bio_id": "itqan:1",
                "source": "itqan",
                "name_ar": bio_only_name,
                "name_ar_normalized": normalize_arabic(bio_only_name),
            }
        ],
    )

    # Stub NER to emit a single mention (into the curated output dir, where
    # disambiguate reads mentions) for the mentioned narrator.
    def _fake_ner(_staging: Path, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            write_narrator_mentions_resolved(
                output_dir,
                [
                    {
                        "mention_id": "m1",
                        "hadith_id": "hs1",
                        "source_corpus": "sunnah",
                        "position_in_chain": 0,
                        "name_raw": mentioned_name,
                        "name_normalized": normalize_arabic(mentioned_name),
                    }
                ],
            )
        ]

    monkeypatch.setattr(ner, "run", _fake_ner)
    # Stub the semantic detector: this test is about bio_promote ordering + the
    # deterministic parallels wiring, not dedup. Left un-stubbed, dedup either
    # requires the ml group or trips an env-specific transformers/torch import —
    # a raise that run_all now surfaces (da#360) instead of swallowing. The
    # deterministic detector below still produces the cross-sect edge asserted.
    monkeypatch.setattr(dedup, "run", lambda *_a, **_k: [])

    run_all(tmp_path / "raw", staging, curated)

    canonical_path = curated / "narrators_canonical.parquet"
    assert canonical_path.exists(), "disambiguate+bio_promote produced no canonical table"
    ids = set(pq.read_table(canonical_path).column("canonical_id").to_pylist())

    # The bio-only narrator survived the disambiguate overwrite ...
    assert bio_only_id in ids, (
        "bio-only narrator was clobbered — bio_promote ran before disambiguate"
    )
    # ... and it coexists with a disambiguate-produced narrator (so the overwrite
    # really happened and the merge landed on top, not a trivial bio-only file).
    assert len(ids) >= 2, "expected disambiguate-produced narrator alongside the promoted bio"

    # detect_parallels is wired and emits the cross-sect edge even with no model.
    links_path = staging / "parallel_links.parquet"
    assert links_path.exists()
    links = pq.read_table(links_path)
    assert links.num_rows >= 1
    assert any(links.column("cross_sect").to_pylist()), "expected a cross-sect PARALLEL_OF edge"
