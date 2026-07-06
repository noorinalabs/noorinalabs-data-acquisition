"""Unit tests for the shared resolve checkpoint primitives (da#272).

Covers the mechanics every resumable resolve stage now shares: atomic
save/load/clear, cadence resolution, the streaming column-group fingerprint, and
the scalar fingerprint — so the stage tests can trust the primitive and only
assert their own state shape.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import _checkpoint


# ---------------------------------------------------------------------------
# dir / save / load / clear
# ---------------------------------------------------------------------------
def test_checkpoint_dir_is_stage_namespaced(tmp_path: Path) -> None:
    assert _checkpoint.checkpoint_dir(tmp_path, "dedup") == tmp_path / ".dedup_checkpoint"
    assert _checkpoint.checkpoint_dir(tmp_path, "parallels") == tmp_path / ".parallels_checkpoint"


def test_save_load_clear_roundtrip_is_atomic(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "disambiguate")
    assert _checkpoint.load_checkpoint(ckpt_dir) is None  # absent → None

    payload = {"schema_version": 1, "processed": 7, "rows": [["a", "b"]]}
    _checkpoint.save_checkpoint(ckpt_dir, payload)
    assert (ckpt_dir / "state.json").exists()
    assert not (ckpt_dir / "state.json.tmp").exists(), "temp file must be renamed away"
    assert _checkpoint.load_checkpoint(ckpt_dir) == payload

    _checkpoint.clear_checkpoint(ckpt_dir)
    assert not ckpt_dir.exists()
    assert _checkpoint.load_checkpoint(ckpt_dir) is None


def test_load_corrupt_state_reads_as_absent(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "dedup")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "state.json").write_text("{not valid json", encoding="utf-8")
    # A torn/corrupt state file must cold-start, not raise.
    assert _checkpoint.load_checkpoint(ckpt_dir) is None


def test_load_non_dict_json_is_none(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "dedup")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "state.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert _checkpoint.load_checkpoint(ckpt_dir) is None


def test_clear_missing_dir_is_noop(tmp_path: Path) -> None:
    _checkpoint.clear_checkpoint(tmp_path / ".never_created")  # must not raise


# ---------------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------------
def test_resolve_cadence_priority_and_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_CADENCE", raising=False)
    # default when neither override nor env
    assert _checkpoint.resolve_cadence(None, "X_CADENCE", 4) == 4
    # override wins over env, floored at 1
    monkeypatch.setenv("X_CADENCE", "9")
    assert _checkpoint.resolve_cadence(2, "X_CADENCE", 4) == 2
    assert _checkpoint.resolve_cadence(0, "X_CADENCE", 4) == 1
    # env used when no override
    assert _checkpoint.resolve_cadence(None, "X_CADENCE", 4) == 9
    # invalid env falls back to default, not a crash
    monkeypatch.setenv("X_CADENCE", "not-an-int")
    assert _checkpoint.resolve_cadence(None, "X_CADENCE", 4) == 4


# ---------------------------------------------------------------------------
# fingerprint — parquet column groups
# ---------------------------------------------------------------------------
def _write(path: Path, ids: list[str], vals: list[str], tags: list[str]) -> None:
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, pa.string()),
                "val": pa.array(vals, pa.string()),
                "tag": pa.array(tags, pa.string()),
            }
        ),
        path,
    )


def test_column_groups_stable_and_isolated(tmp_path: Path) -> None:
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write(a, ["1", "2"], ["x", "y"], ["p", "q"])
    _write(b, ["1", "2"], ["x", "y"], ["p", "q"])  # byte-identical content

    groups = {"content": ("id", "val"), "tag": ("tag",)}
    da, na = _checkpoint.hash_parquet_column_groups(a, groups)
    db, nb = _checkpoint.hash_parquet_column_groups(b, groups)
    assert na == nb == 2
    assert da == db, "identical content must fingerprint identically"

    # A change confined to `tag` flips only the tag digest, not content.
    c = tmp_path / "c.parquet"
    _write(c, ["1", "2"], ["x", "y"], ["p", "DIFFERENT"])
    dc, _ = _checkpoint.hash_parquet_column_groups(c, groups)
    assert dc["content"] == da["content"], "content digest must ignore the tag column"
    assert dc["tag"] != da["tag"], "tag digest must reflect the tag change"


def test_column_group_digest_matches_single_group(tmp_path: Path) -> None:
    """A group's digest is independent of the other groups hashed alongside it."""
    a = tmp_path / "a.parquet"
    _write(a, ["1", "2"], ["x", "y"], ["p", "q"])
    both, _ = _checkpoint.hash_parquet_column_groups(a, {"content": ("id", "val"), "tag": ("tag",)})
    alone, _ = _checkpoint.hash_parquet_column_groups(a, {"content": ("id", "val")})
    assert both["content"] == alone["content"]


def test_missing_input_yields_empty_digests(tmp_path: Path) -> None:
    digests, total = _checkpoint.hash_parquet_column_groups(
        tmp_path / "nope.parquet", {"content": ("id",)}
    )
    assert total == 0
    assert digests == {"content": ""}


def test_multi_path_hashed_in_order(tmp_path: Path) -> None:
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write(a, ["1"], ["x"], ["p"])
    _write(b, ["2"], ["y"], ["q"])
    groups = {"content": ("id", "val")}
    ab, n_ab = _checkpoint.hash_parquet_column_groups([a, b], groups)
    ba, _ = _checkpoint.hash_parquet_column_groups([b, a], groups)
    assert n_ab == 2
    assert ab["content"] != ba["content"], "path order is part of the fingerprint"


# ---------------------------------------------------------------------------
# fingerprint — scalars
# ---------------------------------------------------------------------------
def test_hash_strings_deterministic_and_sensitive() -> None:
    assert _checkpoint.hash_strings("corpus", 0.7, 50) == _checkpoint.hash_strings(
        "corpus", 0.7, 50
    )
    assert _checkpoint.hash_strings("corpus", 0.7, 50) != _checkpoint.hash_strings(
        "corpus", 0.8, 50
    )
    # separator prevents field-boundary collisions ("ab"+"c" != "a"+"bc")
    assert _checkpoint.hash_strings("ab", "c") != _checkpoint.hash_strings("a", "bc")


def test_log_resume_smoke() -> None:
    _checkpoint.log_resume("dedup", skipped=3, total=10, pairs=2)  # must not raise
    _checkpoint.log_resume("parallels", skipped=0, total=0)  # zero-division guard


# ---------------------------------------------------------------------------
# CheckpointController + StopAfterReached (--stop-after, da#276)
# ---------------------------------------------------------------------------
def test_controller_cadence_counting() -> None:
    c = _checkpoint.CheckpointController(3)
    # A checkpoint is due only on the 3rd, 6th, ... completed batch.
    assert [c.batch_complete() for _ in range(6)] == [False, False, True, False, False, True]


def test_controller_no_stop_budget_never_stops() -> None:
    c = _checkpoint.CheckpointController(1)  # stop_after=None
    # checkpoint_written never signals a stop when no budget is set.
    assert [c.checkpoint_written() for _ in range(5)] == [False] * 5
    assert c.checkpoints_written == 5


def test_controller_stop_after_budget() -> None:
    c = _checkpoint.CheckpointController(1, stop_after=2)
    assert c.checkpoint_written() is False  # 1st write
    assert c.checkpoint_written() is True  # 2nd write hits the budget
    assert c.checkpoints_written == 2


def test_controller_cadence_floored_at_one() -> None:
    c = _checkpoint.CheckpointController(0)
    assert c.batch_complete() is True  # cadence floored to 1


def test_controller_stop_logs_and_raises_with_perf() -> None:
    c = _checkpoint.CheckpointController(1, stop_after=1)
    c.checkpoint_written()
    with pytest.raises(_checkpoint.StopAfterReached) as excinfo:
        c.stop("dedup", processed=40, total=200, pairs=7)
    exc = excinfo.value
    assert exc.stage == "dedup"
    assert exc.checkpoints == 1
    assert exc.processed == 40
    assert exc.total == 200
    assert exc.rate_per_s >= 0.0
    summary = exc.summary()
    assert "dedup" in summary and "40/200" in summary and "--from-step dedup" in summary


def test_stop_after_reached_is_baseexception_not_exception() -> None:
    """The load-bearing property: run_all's ``except Exception`` must NOT catch it,
    so the pipeline halts instead of treating the stop as a step failure."""
    exc = _checkpoint.StopAfterReached(
        "parallels", checkpoints=2, processed=4, total=8, elapsed_s=1.0, rate_per_s=4.0
    )
    assert isinstance(exc, BaseException)
    assert not isinstance(exc, Exception)


def test_exit_code_distinct() -> None:
    # 0=completed, 1=crash, 2=argparse/validation — stopped must differ from all.
    assert _checkpoint.EXIT_STOPPED_AT_LIMIT not in (0, 1, 2)
