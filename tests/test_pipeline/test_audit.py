"""Tests for the pipeline audit trail system."""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from src.pipeline.audit import (
    AuditEntry,
    create_audit_entry,
    list_recent_entries,
    write_audit_entry,
)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestAuditEntry:
    def test_create_audit_entry(self) -> None:
        entry = create_audit_entry(
            "load",
            duration_seconds=12.5,
            files_changed=[{"file": "staging/a.parquet", "md5_after": "abc"}],
            rows_affected=100,
            summary={"total_nodes": 100},
        )
        assert entry.stage == "load"
        assert entry.duration_seconds == 12.5
        assert entry.rows_affected == 100
        assert len(entry.files_changed) == 1
        assert entry.summary["total_nodes"] == 100
        assert entry.operator  # should be populated
        assert entry.timestamp  # should be populated


class TestWriteAndRead:
    def test_write_creates_file(self, data_dir: Path) -> None:
        entry = create_audit_entry("sync", duration_seconds=5.0)
        path = write_audit_entry(data_dir, entry)
        assert path.exists()
        assert path.suffix == ".json"
        assert "sync" in path.name

    def test_write_creates_audit_dir(self, data_dir: Path) -> None:
        entry = create_audit_entry("load", duration_seconds=1.0)
        write_audit_entry(data_dir, entry)
        assert (data_dir / "audit").is_dir()

    def test_write_read_only_dir_is_best_effort(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-only filesystem (e.g. the graph-ops container mount) must
        not raise — the audit write is best-effort and must never fail an
        already-completed pipeline stage (da#335)."""

        def _raise_read_only(self: Path, *args: object, **kwargs: object) -> None:
            raise OSError(30, "Read-only file system", str(self))

        monkeypatch.setattr(Path, "mkdir", _raise_read_only)

        entry = create_audit_entry("enrich", duration_seconds=1.0)
        with structlog.testing.capture_logs() as logs:
            result = write_audit_entry(data_dir, entry)

        assert result is None
        assert not (data_dir / "audit").exists()
        assert any(
            log["event"] == "audit_write_failed" and log["log_level"] == "warning" for log in logs
        )

    def test_write_failure_on_file_write_is_best_effort(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same best-effort contract when the dir is writable but the file
        write itself fails (e.g. disk full, permission denied on the file)."""

        def _raise_permission(self: Path, *args: object, **kwargs: object) -> None:
            raise OSError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "write_text", _raise_permission)

        entry = create_audit_entry("enrich", duration_seconds=1.0)
        with structlog.testing.capture_logs() as logs:
            result = write_audit_entry(data_dir, entry)

        assert result is None
        assert any(log["event"] == "audit_write_failed" for log in logs)

    def test_list_recent_entries(self, data_dir: Path) -> None:
        for stage in ("sync", "load", "enrich"):
            entry = create_audit_entry(stage, duration_seconds=1.0)
            write_audit_entry(data_dir, entry)

        entries = list_recent_entries(data_dir, last_n=10)
        assert len(entries) == 3
        # All entries should be AuditEntry instances
        for e in entries:
            assert isinstance(e, AuditEntry)

    def test_list_recent_entries_limit(self, data_dir: Path) -> None:
        for i in range(5):
            entry = create_audit_entry(f"stage{i}", duration_seconds=float(i))
            write_audit_entry(data_dir, entry)

        entries = list_recent_entries(data_dir, last_n=2)
        assert len(entries) == 2

    def test_list_recent_entries_empty_dir(self, data_dir: Path) -> None:
        entries = list_recent_entries(data_dir)
        assert entries == []

    def test_round_trip_preserves_data(self, data_dir: Path) -> None:
        entry = create_audit_entry(
            "load",
            duration_seconds=42.0,
            files_changed=[{"file": "staging/x.parquet", "md5_before": "aaa", "md5_after": "bbb"}],
            rows_affected=500,
            summary={"incremental": True, "files_skipped": 3},
        )
        write_audit_entry(data_dir, entry)

        loaded = list_recent_entries(data_dir, last_n=1)
        assert len(loaded) == 1
        e = loaded[0]
        assert e.stage == "load"
        assert e.duration_seconds == 42.0
        assert e.rows_affected == 500
        assert len(e.files_changed) == 1
        assert e.files_changed[0]["md5_before"] == "aaa"
        assert e.summary["incremental"] is True
