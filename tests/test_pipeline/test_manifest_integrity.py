"""da#350: the last-loaded manifest must describe what was actually loaded.

``_cmd_load`` recorded the manifest of *every* input file as "last loaded" no
matter which stages ran. A ``--nodes-only`` run therefore claimed the edge
inputs had been loaded too. The next ``--incremental`` run diffed against that
claim, found nothing changed, printed "No changes detected. Skipping incremental
load." and returned -- so the edges were **never** loaded, and nothing said so.

The manifest is a change-detection optimisation, so its failure modes are not
symmetric: re-loading something already loaded is free (every loader is
idempotent MERGE), while skipping something never loaded is silent data loss.
Every choice below is biased toward re-loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.cli as cli
import src.graph as graph
from src.graph import LoadSummary
from src.pipeline.manifest import (
    LAST_LOADED_MANIFEST_FILENAME,
    load_manifest,
    save_manifest,
)


class _StubClient:
    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _write_staging_parquet(staging: Path) -> None:
    """A realistic staging file: real columns, real Arabic matn."""
    pq.write_table(
        pa.table(
            {
                "source_id": pa.array(["bukhari:1:1", "bukhari:1:2"], pa.string()),
                "matn_ar": pa.array(
                    [
                        "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى",
                        "بني الإسلام على خمس",
                    ],
                    pa.string(),
                ),
                "collection_id": pa.array(["col:bukhari", "col:bukhari"], pa.string()),
            }
        ),
        staging / "hadiths_bukhari.parquet",
    )


def _write_edge_parquet(staging: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "source_narrator_id": pa.array(["nar:umar"], pa.string()),
                "target_narrator_id": pa.array(["nar:alqama"], pa.string()),
                "hadith_id": pa.array(["hdt:bukhari:1:1"], pa.string()),
            }
        ),
        staging / "network_edges_bukhari.parquet",
    )


class _LoadRecorder:
    """Stands in for ``load_all`` and records whether it was invoked."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> LoadSummary:
        self.calls.append(kwargs)
        return LoadSummary(
            node_results=[],
            edge_results=[],
            validation_results=[],
            total_nodes=2,
            total_edges=1 if not kwargs.get("nodes_only") else 0,
            validation_passed=True,
        )


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _LoadRecorder:
    staging = tmp_path / "staging"
    curated = tmp_path / "curated"
    staging.mkdir()
    curated.mkdir()
    _write_staging_parquet(staging)
    _write_edge_parquet(staging)

    monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
    monkeypatch.setattr(
        "src.config.get_settings",
        lambda: SimpleNamespace(
            data_raw_dir=tmp_path / "raw",
            data_staging_dir=staging,
            data_curated_dir=curated,
        ),
    )
    monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
    recorder = _LoadRecorder()
    monkeypatch.setattr(graph, "load_all", recorder)
    return recorder


class TestNodesOnlyDoesNotPoisonIncremental:
    def test_nodes_only_does_not_record_a_last_loaded_manifest(
        self, loader: _LoadRecorder, tmp_path: Path
    ) -> None:
        cli._cmd_load(nodes_only=True)
        assert not (tmp_path / LAST_LOADED_MANIFEST_FILENAME).exists(), (
            "a nodes-only run loaded no edges; it must not claim the edge inputs were loaded"
        )

    def test_incremental_after_nodes_only_still_loads_edges(
        self, loader: _LoadRecorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The whole defect, end to end.

        On origin/main the nodes-only run writes a full last-loaded manifest, so
        the incremental run finds no changes, returns early, and the edges are
        never loaded.
        """
        cli._cmd_load(nodes_only=True)
        assert len(loader.calls) == 1 and loader.calls[0]["nodes_only"] is True

        cli._cmd_load(incremental=True)

        assert len(loader.calls) == 2, (
            f"incremental after nodes-only short-circuited: {capsys.readouterr().out.strip()!r}"
        )
        assert loader.calls[1]["nodes_only"] is False, "the second run must load edges"

    def test_full_load_does_record_last_loaded(self, loader: _LoadRecorder, tmp_path: Path) -> None:
        """Positive control: the guard above is an observation, not a tautology.

        A complete load *does* write the manifest, so its absence after a
        nodes-only run is a real difference in behaviour.
        """
        cli._cmd_load()
        assert (tmp_path / LAST_LOADED_MANIFEST_FILENAME).exists()

    def test_second_full_load_with_no_changes_is_skipped(
        self, loader: _LoadRecorder, tmp_path: Path
    ) -> None:
        """The optimisation still works after a genuine full load."""
        cli._cmd_load()
        cli._cmd_load(incremental=True)
        assert len(loader.calls) == 1, "unchanged inputs after a full load should skip"


class TestManifestWriteIsAtomic:
    def test_save_manifest_leaves_no_partial_file_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-write must not leave a half-written manifest behind."""
        target = tmp_path / LAST_LOADED_MANIFEST_FILENAME
        target.write_text('{"staging/a.parquet": {"md5": "old"}}\n')

        real_replace = Path.replace

        def _boom(self: Path, other: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "replace", _boom)
        assert save_manifest({"staging/b.parquet": {"md5": "new"}}, target) is None
        monkeypatch.setattr(Path, "replace", real_replace)

        # The previous manifest survives intact and no debris is left in the dir.
        assert json.loads(target.read_text()) == {"staging/a.parquet": {"md5": "old"}}
        assert list(tmp_path.iterdir()) == [target]


class TestCorruptManifestFailsSafe:
    def test_corrupt_manifest_reads_as_empty_forcing_a_full_load(self, tmp_path: Path) -> None:
        """Unparseable manifest => reload everything, never skip anything.

        Returning ``{}`` makes ``compare_manifests`` classify every input as
        *added*, so the next load is a full one. The opposite bias -- trusting a
        damaged file -- would silently skip inputs.
        """
        bad = tmp_path / LAST_LOADED_MANIFEST_FILENAME
        bad.write_text('{"staging/a.parquet": {"md5": "trunc')  # truncated JSON
        assert load_manifest(bad) == {}
