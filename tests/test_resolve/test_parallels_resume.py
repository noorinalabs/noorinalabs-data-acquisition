"""Crash-resume tests for ``src.resolve.parallels`` (da#272).

The anchor scan is the long part on the full corpus; these assert the checkpoint
contract that makes it recoverable: a resumed scan is output-identical to a cold
one, a stale-fingerprint checkpoint is discarded, and ``resume=False`` cold-starts.
Fixtures are tiny synthetic ``hadiths_*.parquet`` files — never the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import _checkpoint, parallels


def _write_hadiths(staging: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write one ``hadiths_test.parquet`` with the columns the detector reads."""
    pq.write_table(
        pa.table(
            {
                "source_id": pa.array([r[0] for r in rows], pa.string()),
                "matn_ar": pa.array([None] * len(rows), pa.string()),
                "matn_en": pa.array([r[1] for r in rows], pa.string()),
                "sect": pa.array([r[2] for r in rows], pa.string()),
            }
        ),
        staging / "hadiths_test.parquet",
    )


def _corpus(n_pairs: int) -> list[tuple[str, str, str]]:
    """`n_pairs` exact-duplicate hadith pairs, each a sunni+shia cross-sect match.

    Members of a pair share an identical matn (Jaccard 1.0); different pairs share
    no tokens (Jaccard 0.0), so exactly `n_pairs` cross-sect links are emitted and
    the scan order of anchors is what a resume must reproduce.
    """
    rows: list[tuple[str, str, str]] = []
    for p in range(n_pairs):
        matn = f"alpha{p} beta{p} gamma{p} delta{p}"
        rows.append((f"h{2 * p}", matn, "sunni"))
        rows.append((f"h{2 * p + 1}", matn, "shia"))
    return rows


def _links(staging: Path) -> list[dict[str, object]]:
    return pq.read_table(staging / "parallel_links.parquet").to_pylist()


def _setup(tmp_path: Path, name: str, *, n_pairs: int = 4) -> Path:
    staging = tmp_path / name
    staging.mkdir(parents=True)
    _write_hadiths(staging, _corpus(n_pairs))
    return staging


def test_cold_run_finds_cross_sect_pairs(tmp_path: Path) -> None:
    staging = _setup(tmp_path, "cold")
    parallels.detect_parallels(staging)
    links = _links(staging)
    assert len(links) == 4
    assert all(row["cross_sect"] for row in links)
    assert not _checkpoint.checkpoint_dir(staging, "parallels").exists(), (
        "success clears checkpoint"
    )


def test_resumed_scan_is_output_identical_to_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cold = _setup(tmp_path, "cold")
    parallels.detect_parallels(cold, checkpoint_every_n_blocks=1)
    cold_links = _links(cold)

    resume = _setup(tmp_path, "resume")
    # Checkpoint after every single anchor so a mid-scan crash leaves a partial one.
    monkeypatch.setattr(parallels, "_PARALLELS_ANCHORS_PER_BLOCK", 1)

    real_partners = parallels._candidate_partners

    def crashing_partners(i: int, *args: object, **kwargs: object) -> set[int]:
        if i == 5:
            raise RuntimeError("simulated host crash mid-parallels-scan")
        return real_partners(i, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(parallels, "_candidate_partners", crashing_partners)
    with pytest.raises(RuntimeError):
        parallels.detect_parallels(resume, checkpoint_every_n_blocks=1)
    monkeypatch.setattr(parallels, "_candidate_partners", real_partners)

    ckpt = _checkpoint.load_checkpoint(_checkpoint.checkpoint_dir(resume, "parallels"))
    assert ckpt is not None
    assert 0 < ckpt["processed_anchors"] < 8, "a partial checkpoint must survive the crash"

    # Resume to completion — byte-identical links + order, checkpoint cleared.
    parallels.detect_parallels(resume, checkpoint_every_n_blocks=1)
    assert _links(resume) == cold_links
    assert not _checkpoint.checkpoint_dir(resume, "parallels").exists()


def test_stale_fingerprint_cold_starts(tmp_path: Path) -> None:
    staging = _setup(tmp_path, "stale")
    ckpt_dir = _checkpoint.checkpoint_dir(staging, "parallels")
    _checkpoint.save_checkpoint(
        ckpt_dir,
        {
            "schema_version": parallels._PARALLELS_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": "STALE",
            "processed_anchors": 3,
            "cross_sect_count": 1,
            "records": [
                {
                    "hadith_id_a": "SENTINEL_A",
                    "hadith_id_b": "SENTINEL_B",
                    "similarity_score": 1.0,
                    "variant_type": "verbatim",
                    "cross_sect": True,
                }
            ],
        },
    )
    parallels.detect_parallels(staging)
    ids = {row["hadith_id_a"] for row in _links(staging)}
    assert "SENTINEL_A" not in ids, "stale checkpoint must be discarded (cold start)"


def test_resume_false_ignores_checkpoint(tmp_path: Path) -> None:
    staging = _setup(tmp_path, "noresume")
    # Fingerprint-VALID checkpoint (would otherwise resume), carrying a sentinel.
    hadith_files = sorted(staging.glob("**/hadiths_*.parquet"))
    digests, _ = _checkpoint.hash_parquet_column_groups(
        hadith_files, {"content": parallels._FINGERPRINT_HADITH_COLS}
    )
    # Mirror detect_parallels' fingerprint construction for the default params.
    from src.resolve.parallels import _DEFAULT_MAX_BLOCK_DF, _DEFAULT_THRESHOLD, _load_hadith_rows

    n_indexed = sum(
        1
        for r in _load_hadith_rows(staging)
        if isinstance(r.get("source_id"), str)
        and r["source_id"]
        and parallels._tokenize(
            r.get("matn_ar") if isinstance(r.get("matn_ar"), str) else None,
            r.get("matn_en") if isinstance(r.get("matn_en"), str) else None,
        )
    )
    fingerprint = (
        f"{digests['content']}:{round(_DEFAULT_THRESHOLD, 6)}:{_DEFAULT_MAX_BLOCK_DF}:{n_indexed}"
    )
    ckpt_dir = _checkpoint.checkpoint_dir(staging, "parallels")
    _checkpoint.save_checkpoint(
        ckpt_dir,
        {
            "schema_version": parallels._PARALLELS_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "processed_anchors": 8,
            "cross_sect_count": 1,
            "records": [
                {
                    "hadith_id_a": "SENTINEL_A",
                    "hadith_id_b": "SENTINEL_B",
                    "similarity_score": 1.0,
                    "variant_type": "verbatim",
                    "cross_sect": True,
                }
            ],
        },
    )
    parallels.detect_parallels(staging, resume=False)
    ids = {row["hadith_id_a"] for row in _links(staging)}
    assert "SENTINEL_A" not in ids, "resume=False must ignore even a valid checkpoint"
