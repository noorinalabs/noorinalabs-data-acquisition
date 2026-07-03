"""``resolve --stop-after N`` bounded partial-run tests (da#276).

The flag stops a resumable stage cleanly after its Nth checkpoint write, leaving
the checkpoint on disk and writing NO final output, then halts the pipeline with a
distinguishable exit status. The load-bearing guarantee: a bounded stop followed
by a bare resume equals an uninterrupted run on the covered prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.cli as cli
from src.resolve import EXIT_STOPPED_AT_LIMIT, StopAfterReached, _checkpoint, parallels
from src.resolve.dedup import _search_and_collect_resumable


# ---------------------------------------------------------------------------
# parallels: stop-after then resume == uninterrupted (the must-have)
# ---------------------------------------------------------------------------
def _write_hadiths(staging: Path, n_pairs: int) -> None:
    rows: list[tuple[str, str, str]] = []
    for p in range(n_pairs):
        matn = f"alpha{p} beta{p} gamma{p} delta{p}"
        rows.append((f"h{2 * p}", matn, "sunni"))
        rows.append((f"h{2 * p + 1}", matn, "shia"))
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


def _links(staging: Path) -> list[dict[str, object]]:
    return pq.read_table(staging / "parallel_links.parquet").to_pylist()


def test_parallels_stop_leaves_checkpoint_and_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "s"
    staging.mkdir()
    _write_hadiths(staging, n_pairs=4)  # 8 anchors
    monkeypatch.setattr(parallels, "_PARALLELS_ANCHORS_PER_BLOCK", 1)

    with pytest.raises(StopAfterReached) as excinfo:
        parallels.detect_parallels(staging, checkpoint_every_n_blocks=1, stop_after=2)

    assert excinfo.value.stage == "parallels"
    # Checkpoint left on disk; final output NOT written.
    ckpt = _checkpoint.load_checkpoint(_checkpoint.checkpoint_dir(staging, "parallels"))
    assert ckpt is not None
    assert ckpt["processed_anchors"] == 2
    assert not (staging / "parallel_links.parquet").exists(), "no partial output on stop"


def test_parallels_stop_then_resume_equals_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Uninterrupted reference.
    cold = tmp_path / "cold"
    cold.mkdir()
    _write_hadiths(cold, n_pairs=4)
    parallels.detect_parallels(cold)
    cold_links = _links(cold)

    # Bounded stop, then a bare resume to completion.
    probe = tmp_path / "probe"
    probe.mkdir()
    _write_hadiths(probe, n_pairs=4)
    monkeypatch.setattr(parallels, "_PARALLELS_ANCHORS_PER_BLOCK", 1)
    with pytest.raises(StopAfterReached):
        parallels.detect_parallels(probe, checkpoint_every_n_blocks=1, stop_after=2)
    # Resume (no stop budget) — finishes from checkpoint 2.
    parallels.detect_parallels(probe, checkpoint_every_n_blocks=1)

    assert _links(probe) == cold_links
    assert not _checkpoint.checkpoint_dir(probe, "parallels").exists(), "resume clears checkpoint"


# ---------------------------------------------------------------------------
# dedup: stop-after then resume == uninterrupted (numpy-only fake index)
# ---------------------------------------------------------------------------
_EMB = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
_IDS = [f"h{i}" for i in range(6)]
_CORPUS = {f"h{i}": ("sunnah" if i % 2 == 0 else "thaqalayn") for i in range(6)}


class _FakeIndex:
    def __init__(self, emb: npt.NDArray[np.float32]) -> None:
        self.emb = emb

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        sims = queries @ self.emb.T
        idx = np.argsort(-sims, axis=1)[:, :k].astype(np.int64)
        return np.take_along_axis(sims, idx, axis=1).astype(np.float32), idx


def _collect(ckpt_dir: Path, *, resume: bool, index_available: bool, stop_after: int | None):
    idx = _FakeIndex(_EMB)
    return _search_and_collect_resumable(
        embeddings=_EMB,
        hadith_ids=_IDS,
        id_to_corpus=_CORPUS,
        actual_k=6,
        threshold=0.5,
        ckpt_dir=ckpt_dir,
        fingerprint="fp",
        cadence=1,
        resume=resume,
        index_available=index_available,
        build_index=lambda: idx,
        reload_index=lambda: idx,
        block_size=1,  # one row per block ⇒ a checkpoint per row
        stop_after=stop_after,
    )


def test_dedup_stop_then_resume_equals_uninterrupted(tmp_path: Path) -> None:
    cold = _collect(
        _checkpoint.checkpoint_dir(tmp_path / "cold", "dedup"),
        resume=True,
        index_available=False,
        stop_after=None,
    )

    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path / "probe", "dedup")
    with pytest.raises(StopAfterReached) as excinfo:
        _collect(ckpt_dir, resume=True, index_available=False, stop_after=2)
    assert excinfo.value.stage == "dedup"
    saved = _checkpoint.load_checkpoint(ckpt_dir)
    assert saved is not None and saved["processed_rows"] == 2  # 2 blocks/rows done

    resumed = _collect(ckpt_dir, resume=True, index_available=True, stop_after=None)
    assert resumed == cold


# ---------------------------------------------------------------------------
# CLI: exit-code mapping + argument validation
# ---------------------------------------------------------------------------
def test_cli_maps_stop_to_distinct_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Settings:
        data_raw_dir = "/x"
        data_staging_dir = "/y"
        data_curated_dir = "/z"

    def _raise(*_a: object, **_k: object) -> object:
        raise StopAfterReached(
            "dedup", checkpoints=3, processed=30, total=100, elapsed_s=2.0, rate_per_s=15.0
        )

    monkeypatch.setattr("src.config.get_settings", lambda: _Settings())
    monkeypatch.setattr("src.resolve.run_all", _raise)

    with pytest.raises(SystemExit) as excinfo:
        cli._cmd_resolve(from_step="dedup", stop_after=3)
    assert excinfo.value.code == EXIT_STOPPED_AT_LIMIT


def test_cli_completion_is_not_a_stopped_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other side of the exit-code contract: a normal completion must NOT take
    the stopped-at-limit exit — stopped (3) is distinct from completed."""

    class _Settings:
        data_raw_dir = "/x"
        data_staging_dir = "/y"
        data_curated_dir = "/z"

    monkeypatch.setattr("src.config.get_settings", lambda: _Settings())
    monkeypatch.setattr("src.resolve.run_all", lambda *_a, **_k: {"dedup": []})
    # Returns normally (prints "Resolution complete."), never raises SystemExit.
    cli._cmd_resolve(stop_after=None)


@pytest.mark.parametrize(
    "argv",
    [
        ["prog", "resolve", "--from-step", "ner", "--stop-after", "3"],
        ["prog", "resolve", "--from-step", "bio_promote", "--stop-after", "3"],
        ["prog", "resolve", "--stop-after", "0"],
    ],
)
def test_cli_rejects_invalid_stop_after(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2, "argparse validation error exit"


def test_cli_valid_combination_threads_all_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_cmd_resolve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["prog", "resolve", "--from-step", "dedup", "--no-resume", "--stop-after", "5"]
    )
    cli.main()
    assert captured == {"from_step": "dedup", "resume": False, "stop_after": 5}
