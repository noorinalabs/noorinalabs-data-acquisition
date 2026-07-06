"""Crash-resume tests for fuzzy_cluster's block-scoring pass (da#272, PR2).

The multi-day cluster pass checkpoints the union-find + applied-block set. These
assert the contract: a ``--stop-after`` bounded run then a bare resume produces
the same merged canonical table as an uninterrupted run, a stale-fingerprint
checkpoint is discarded, ``--no-resume`` cold-starts, and the pure ``cluster_records``
callers never touch a checkpoint. Fixtures are tiny synthetic canonical tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.parse.identity import make_canonical_id
from src.resolve import _checkpoint, fuzzy_cluster
from src.resolve._checkpoint import StopAfterReached
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.arabic import normalize_arabic


def _rec(name: str, **over: Any) -> dict[str, Any]:
    norm = normalize_arabic(name)
    base: dict[str, Any] = {f.name: None for f in NARRATORS_CANONICAL_SCHEMA}
    base.update(
        {
            "canonical_id": make_canonical_id(norm),
            "name_ar": name,
            "name_ar_normalized": norm,
            "aliases": [],
            "source_ids": [],
            "source_corpora": [],
            "mention_count": 1,
        }
    )
    base.update(over)
    return base


def _input_records() -> list[dict[str, Any]]:
    """Four independent variant pairs (each a fuller-nisba + shorter form of one
    scholar), so the block-scoring pass has four separate blocks to apply — enough
    to stop partway and resume."""
    pairs = [
        ("احمد بن حنبل الشيباني", "احمد بن حنبل", 241),
        ("محمد بن اسماعيل البخاري", "محمد بن اسماعيل", 256),
        ("مالك بن انس المدني", "مالك بن انس", 179),
        ("سفيان بن عيينة الهلالي", "سفيان بن عيينة", 198),
    ]
    records: list[dict[str, Any]] = []
    for full, short, death in pairs:
        records.append(_rec(full, source_corpora=["itqan"], death_year_ah=death))
        records.append(_rec(short, source_corpora=["sanadset"], death_year_ah=death))
    return records


def _write_canonical(path: Path, records: list[dict[str, Any]]) -> None:
    arrays = {f.name: [r.get(f.name) for r in records] for f in NARRATORS_CANONICAL_SCHEMA}
    pq.write_table(pa.table(arrays, schema=NARRATORS_CANONICAL_SCHEMA), path)


def _output_rows(path: Path) -> list[dict[str, Any]]:
    """Merged canonical rows, sorted by survivor id (cluster order is union-order-
    dependent but the survivor SET is invariant)."""
    rows = pq.read_table(path).to_pylist()
    return sorted(rows, key=lambda r: r["canonical_id"])


def _setup(tmp_path: Path, name: str) -> tuple[Path, Path]:
    d = tmp_path / name
    d.mkdir(parents=True)
    canonical = d / "narrators_canonical.parquet"
    _write_canonical(canonical, _input_records())
    return d, canonical


def test_cold_run_merges_all_pairs(tmp_path: Path) -> None:
    staging, canonical = _setup(tmp_path, "cold")
    metrics = fuzzy_cluster.cluster_canonical_narrators(canonical, staging_dir=staging)
    assert metrics.merged_records == 4  # 8 records → 4 survivors
    assert len(_output_rows(canonical)) == 4
    assert not _checkpoint.checkpoint_dir(staging, "cluster").exists(), "success clears checkpoint"


def test_stop_leaves_checkpoint_and_does_not_rewrite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, canonical = _setup(tmp_path, "stop")
    monkeypatch.setattr(fuzzy_cluster, "_CLUSTER_CHECKPOINT_SCORED_INTERVAL", 1)

    with pytest.raises(StopAfterReached) as excinfo:
        fuzzy_cluster.cluster_canonical_narrators(canonical, staging_dir=staging, stop_after=2)

    assert excinfo.value.stage == "cluster"
    ckpt = _checkpoint.load_checkpoint(_checkpoint.checkpoint_dir(staging, "cluster"))
    assert ckpt is not None
    assert len(ckpt["parent"]) == 8
    # Output NOT rewritten: canonical still holds the 8 input rows.
    assert len(pq.read_table(canonical).to_pylist()) == 8


def test_stop_then_resume_equals_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Uninterrupted reference.
    cold_staging, cold_canonical = _setup(tmp_path, "cold")
    fuzzy_cluster.cluster_canonical_narrators(cold_canonical, staging_dir=cold_staging)
    cold_rows = _output_rows(cold_canonical)

    # Bounded stop, then a bare resume to completion.
    probe_staging, probe_canonical = _setup(tmp_path, "probe")
    monkeypatch.setattr(fuzzy_cluster, "_CLUSTER_CHECKPOINT_SCORED_INTERVAL", 1)
    with pytest.raises(StopAfterReached):
        fuzzy_cluster.cluster_canonical_narrators(
            probe_canonical, staging_dir=probe_staging, stop_after=2
        )
    fuzzy_cluster.cluster_canonical_narrators(probe_canonical, staging_dir=probe_staging)

    assert _output_rows(probe_canonical) == cold_rows
    assert not _checkpoint.checkpoint_dir(probe_staging, "cluster").exists()


def test_stop_then_resume_equals_uninterrupted_pooled_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same equivalence, but forcing the CONCURRENT cdist/futures path (not the
    scalar inline path the tiny fixtures otherwise take). This is the path the
    real multi-day run uses, where block_idx is tracked through the ``pending``
    dict and applied in thread-completion order — the checkpoint's riskiest code.
    """
    # _SCALAR_BLOCK_MAX=1 sends every m>=2 block to the pool + cdist worker.
    monkeypatch.setattr(fuzzy_cluster, "_SCALAR_BLOCK_MAX", 1)

    cold_staging, cold_canonical = _setup(tmp_path, "cold_pool")
    fuzzy_cluster.cluster_canonical_narrators(cold_canonical, staging_dir=cold_staging)
    cold_rows = _output_rows(cold_canonical)
    assert len(cold_rows) == 4  # the pooled path finds the same 4 merges

    probe_staging, probe_canonical = _setup(tmp_path, "probe_pool")
    monkeypatch.setattr(fuzzy_cluster, "_CLUSTER_CHECKPOINT_SCORED_INTERVAL", 1)
    with pytest.raises(StopAfterReached):
        fuzzy_cluster.cluster_canonical_narrators(
            probe_canonical, staging_dir=probe_staging, stop_after=2
        )
    # A partial checkpoint from the pooled path survived, with fewer than all
    # blocks applied.
    ckpt = _checkpoint.load_checkpoint(_checkpoint.checkpoint_dir(probe_staging, "cluster"))
    assert ckpt is not None and 0 < len(ckpt["applied_blocks"]) < 4

    fuzzy_cluster.cluster_canonical_narrators(probe_canonical, staging_dir=probe_staging)
    assert _output_rows(probe_canonical) == cold_rows
    assert not _checkpoint.checkpoint_dir(probe_staging, "cluster").exists()


def test_stale_fingerprint_cold_starts(tmp_path: Path) -> None:
    staging, canonical = _setup(tmp_path, "stale")
    # A stale checkpoint whose union-find would collapse everything into ONE cluster
    # if wrongly reused (parent all-0). Length matches n so only the fingerprint
    # mismatch can trigger the discard.
    _checkpoint.save_checkpoint(
        _checkpoint.checkpoint_dir(staging, "cluster"),
        {
            "schema_version": fuzzy_cluster._CLUSTER_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": "STALE",
            "parent": [0] * 8,
            "applied_blocks": [0, 1, 2, 3, 4, 5, 6, 7],
            "scored": 999,
            "merged": 7,
        },
    )
    metrics = fuzzy_cluster.cluster_canonical_narrators(canonical, staging_dir=staging)
    assert metrics.merged_records == 4, "stale checkpoint must be discarded, not collapse to 1"


def test_cap_change_across_resume_discards_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume across a da#270 cap change must DISCARD the checkpoint (da#303).

    The caps (`_MAX_MATCH_KEYS_PER_RECORD` / `_MAX_BLOCKING_TOKENS_PER_RECORD`)
    shape the block universe, and the skip-set is POSITIONAL block indices, so
    accepting a checkpoint taken at a different cap would restore an old-universe
    union-find against a new-universe block list — silent corruption. This is the
    true discriminator: it fails if the caps are absent from the fingerprint.
    """
    staging, canonical = _setup(tmp_path, "capchange")
    monkeypatch.setattr(fuzzy_cluster, "_CLUSTER_CHECKPOINT_SCORED_INTERVAL", 1)
    # A real partial checkpoint at the default caps, carrying non-trivial merges.
    with pytest.raises(StopAfterReached):
        fuzzy_cluster.cluster_canonical_narrators(canonical, staging_dir=staging, stop_after=1)
    ckpt = _checkpoint.load_checkpoint(_checkpoint.checkpoint_dir(staging, "cluster"))
    assert ckpt is not None and ckpt["merged"] > 0

    # Collapse the blocking-token cap to 1 → no composite key can form, so the NEW
    # universe has zero candidate pairs and the correct cold result is 0 merges. A
    # wrongly-accepted checkpoint would instead surface the old-cap merges (>0).
    monkeypatch.setattr(fuzzy_cluster, "_MAX_BLOCKING_TOKENS_PER_RECORD", 1)
    metrics = fuzzy_cluster.cluster_canonical_narrators(canonical, staging_dir=staging)
    assert metrics.merged_records == 0, "cap change must discard the checkpoint (cold re-cluster)"


def test_no_resume_ignores_checkpoint(tmp_path: Path) -> None:
    staging, canonical = _setup(tmp_path, "noresume")
    _checkpoint.save_checkpoint(
        _checkpoint.checkpoint_dir(staging, "cluster"),
        {
            "schema_version": fuzzy_cluster._CLUSTER_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": "anything",
            "parent": [0] * 8,  # would collapse to 1 cluster if used
            "applied_blocks": list(range(8)),
            "scored": 999,
            "merged": 7,
        },
    )
    metrics = fuzzy_cluster.cluster_canonical_narrators(
        canonical, staging_dir=staging, resume=False
    )
    assert metrics.merged_records == 4, "resume=False must cold-start"


def test_pure_cluster_records_writes_no_checkpoint(tmp_path: Path) -> None:
    # cluster_canonical_narrators WITHOUT staging_dir (and the pure cluster_records
    # path used by cluster_assignment / the quality harness) must not checkpoint.
    _, canonical = _setup(tmp_path, "pure")
    fuzzy_cluster.cluster_canonical_narrators(canonical)  # no staging_dir
    assert not (tmp_path / "pure" / ".cluster_checkpoint").exists()
    # And the pure records API produces the same 4 clusters with no IO.
    clusters = fuzzy_cluster.cluster_records(_input_records())
    assert sorted(len(c) for c in clusters) == [2, 2, 2, 2]
