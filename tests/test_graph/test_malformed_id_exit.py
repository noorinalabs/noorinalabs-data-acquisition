"""da#359: a load that quarantines malformed ids must not exit 0.

da#355 made the loaders quarantine a doubled-corpus id rather than repair it.
Quarantining is right -- these loaders commit per batch with no spanning
transaction, so an escaping exception strands batches 1..N-1 in Neo4j hours
into a load. But quarantine terminating in a zero exit turned a silent repair
into a silent drop:

    650,986 of 853,218 staging hadiths (76.3%) carry a ``sanadset:sanadset:``
    id. Every one raises at ``hadith_node_id``. ``cli._cmd_load`` passes
    ``strict=False``, so every one is quarantined -- and the load reported
    success, validation reported "all within threshold", and three quarters of
    the corpus was absent from the graph.

Three gates were blind. ``parse_all`` swallows the producer-side grammar
assertion (da#380); ``collection_coverage`` cannot measure a collection with a
NULL ``expected_count``, which is 67 of 68 (da#382). The load-side exit code is
therefore the one gate that has to carry the weight, so it is tested here
against the id the production parquet actually contains.

The fixture id is copied verbatim from ``data/staging/hadiths_sanadset.parquet``
row 0, and the file carries the production filename. A fixture that cannot
produce the failing state proves nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli as cli
import src.graph as graph
from src.graph import EXIT_MALFORMED_IDS, EdgeLoadResult, LoadResult, LoadSummary
from src.graph.load_edges import _load_appears_in, _load_graded_by
from src.graph.load_nodes import _load_hadiths
from src.graph.validate import ValidationResult
from src.resolve._checkpoint import EXIT_STOPPED_AT_LIMIT

from .conftest import MockNeo4jClient, write_hadiths

# Verbatim from data/staging/hadiths_sanadset.parquet, row 0. All 650,986 rows
# share the `sanadset:sanadset` prefix. Do NOT replace this with a synthetic
# string: the point of the test is that the shape the producer really emits
# reaches `hadith_node_id` and raises.
REAL_DOUBLED_SOURCE_ID = "sanadset:sanadset:0:0:0"

# Same corpus, same file, same schema -- differing ONLY in the second segment.
# This isolates the doubling as the cause rather than the corpus.
CONTROL_SOURCE_ID = "sanadset:bukhari:1"


def _sanadset_row(source_id: str, collection_name: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_corpus": "sanadset",
        "collection_name": collection_name,
        "matn_ar": f"متن {source_id}",
        "sect": "sunni",
    }


class TestLoaderCountsMalformedIdsSeparately:
    """`skipped` already means three different things. Malformed needs its own."""

    def test_real_sanadset_id_is_counted_as_malformed(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        # Production filename: hadiths_sanadset.parquet.
        write_hadiths(
            staging,
            [
                _sanadset_row(REAL_DOUBLED_SOURCE_ID, "sanadset"),
                _sanadset_row(CONTROL_SOURCE_ID, "bukhari"),
            ],
            suffix="sanadset",
        )

        result = _load_hadiths(MockNeo4jClient(), staging, strict=False)

        # The malformed row is counted as malformed, by its own name.
        assert result.malformed_ids == 1
        # POSITIVE CONTROL: the well-formed row in the same file still loaded.
        # Without this, `malformed_ids == 1` would also hold if the loader had
        # simply refused the whole file.
        assert result.created == 1

    def test_wellformed_file_reports_zero_malformed(self, tmp_path: Path) -> None:
        """Makes the assertion above a measurement rather than a constant."""
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(staging, [_sanadset_row(CONTROL_SOURCE_ID, "bukhari")], suffix="sanadset")

        result = _load_hadiths(MockNeo4jClient(), staging, strict=False)

        assert result.malformed_ids == 0
        assert result.created == 1

    def test_malformed_is_not_an_alias_of_skipped(self, tmp_path: Path) -> None:
        """`skipped` already counts non-canonical drops. `malformed_ids` must not.

        `fawaz:bukhari` is non-canonical (deduplicated to the `lk` spine) and is
        skipped for a reason that is not a grammar violation. A file carrying
        both causes must report `skipped=2, malformed_ids=1` -- if the two
        counters moved together, a test asserting a nonzero skip count would
        pass over a broken fix.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        noncanonical = {
            "source_id": "fawaz:bukhari:1",
            "source_corpus": "fawaz",
            "collection_name": "bukhari",
            "matn_ar": "متن fawaz",
            "sect": "sunni",
        }
        write_hadiths(
            staging,
            [
                noncanonical,
                _sanadset_row(REAL_DOUBLED_SOURCE_ID, "sanadset"),
                _sanadset_row(CONTROL_SOURCE_ID, "bukhari"),
            ],
            suffix="sanadset",
        )

        result = _load_hadiths(MockNeo4jClient(), staging, strict=False)

        assert result.skipped == 2
        assert result.malformed_ids == 1
        assert result.created == 1


class TestEdgeLoadersReportMalformedOnTheEmptyBatchPath:
    """The all-malformed load takes an EARLY return, and it must carry the count.

    This is not a corner case, it is the production case: every one of the
    650,986 sanadset rows is malformed, so the edge loaders quarantine all of
    them, `batch` is empty, and they return through `if not batch:` -- never
    reaching the normal return. A count wired only into the normal return
    reports `malformed_ids=0` for the exact load that dropped everything.

    Caught by mutation: deleting `malformed_ids=` from `_load_graded_by`'s
    empty-batch return left the whole 183-test suite green.
    """

    def _all_malformed_staging(self, tmp_path: Path) -> Path:
        staging = tmp_path / "staging"
        staging.mkdir()
        rows = []
        for row in (
            _sanadset_row(REAL_DOUBLED_SOURCE_ID, "sanadset"),
            _sanadset_row("sanadset:sanadset:0:0:1", "sanadset"),
        ):
            row["grade"] = "sahih"  # GRADED_BY only considers graded rows
            rows.append(row)
        write_hadiths(staging, rows, suffix="sanadset")
        return staging

    def test_graded_by_empty_batch_still_reports_malformed(self, tmp_path: Path) -> None:
        staging = self._all_malformed_staging(tmp_path)

        result = _load_graded_by(MockNeo4jClient(), staging, strict=False)

        assert result.created == 0  # nothing loaded: the empty-batch path
        assert result.malformed_ids == 2

    def test_appears_in_empty_batch_still_reports_malformed(self, tmp_path: Path) -> None:
        staging = self._all_malformed_staging(tmp_path)

        result = _load_appears_in(MockNeo4jClient(), staging, strict=False)

        assert result.created == 0
        assert result.malformed_ids == 2

    def test_a_load_of_only_malformed_rows_exits_nonzero(self, tmp_path: Path) -> None:
        """End to end over the loaders: every gate reports, and the total is nonzero."""
        staging = self._all_malformed_staging(tmp_path)
        summary = LoadSummary(
            node_results=[_load_hadiths(MockNeo4jClient(), staging, strict=False)],
            edge_results=[
                _load_graded_by(MockNeo4jClient(), staging, strict=False),
                _load_appears_in(MockNeo4jClient(), staging, strict=False),
            ],
        )
        assert summary.total_nodes == 0
        assert summary.total_malformed_ids == 6  # 2 hadith + 2 graded_by + 2 appears_in


class TestLoadSummaryAggregation:
    def test_total_counts_both_node_and_edge_results(self) -> None:
        """An aggregation that summed only `node_results` would pass a weaker test."""
        summary = LoadSummary(
            node_results=[LoadResult("Hadith", 0, 0, 7, malformed_ids=7)],
            edge_results=[EdgeLoadResult("GRADED_BY", 0, 3, 0, malformed_ids=3)],
        )
        assert summary.total_malformed_ids == 10

    def test_total_is_zero_on_a_clean_load(self) -> None:
        summary = LoadSummary(
            node_results=[LoadResult("Hadith", 5, 0, 0)],
            edge_results=[EdgeLoadResult("GRADED_BY", 5, 0, 0)],
        )
        assert summary.total_malformed_ids == 0


class _StubClient:
    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _install_load_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, summary: LoadSummary
) -> None:
    """Neutralise everything in `_cmd_load` except the exit-code decision."""
    (tmp_path / "staging").mkdir(exist_ok=True)
    (tmp_path / "curated").mkdir(exist_ok=True)

    monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
    monkeypatch.setattr(
        "src.config.get_settings",
        lambda: SimpleNamespace(
            data_raw_dir=tmp_path / "raw",
            data_staging_dir=tmp_path / "staging",
            data_curated_dir=tmp_path / "curated",
        ),
    )
    monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", _StubClient)
    monkeypatch.setattr(graph, "load_all", lambda *a, **k: summary)


def _summary_from_the_real_loader(tmp_path: Path) -> LoadSummary:
    """Build the summary by running the REAL loader over the REAL doubled id.

    The malformed count is therefore an observation of production code over a
    production-shaped row, not a number typed into a test.
    """
    staging = tmp_path / "real_staging"
    staging.mkdir()
    write_hadiths(
        staging,
        [
            _sanadset_row(REAL_DOUBLED_SOURCE_ID, "sanadset"),
            _sanadset_row(CONTROL_SOURCE_ID, "bukhari"),
        ],
        suffix="sanadset",
    )
    hadiths = _load_hadiths(MockNeo4jClient(), staging, strict=False)
    assert hadiths.malformed_ids == 1, "fixture no longer produces the failing state"
    return LoadSummary(node_results=[hadiths], edge_results=[], total_nodes=hadiths.created)


class TestCmdLoadExitCode:
    def test_quarantined_ids_exit_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: a load that refused input must not report success."""
        _install_load_stubs(monkeypatch, tmp_path, _summary_from_the_real_loader(tmp_path))

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load(skip_validation=True)

        assert exc.value.code == EXIT_MALFORMED_IDS

    def test_clean_load_still_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSITIVE CONTROL: the exit is conditional on the counter, not unconditional."""
        clean = LoadSummary(node_results=[LoadResult("Hadith", 5, 0, 0)], total_nodes=5)
        _install_load_stubs(monkeypatch, tmp_path, clean)

        cli._cmd_load(skip_validation=True)  # falls off the end -> rc 0

    def test_fires_even_when_validation_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`skip_validation` and `nodes_only` bypass the only other non-zero exit.

        The malformed check hangs off the load, not off the validation block, so
        `--nodes-only` cannot launder a quarantining load into a green run.
        """
        _install_load_stubs(monkeypatch, tmp_path, _summary_from_the_real_loader(tmp_path))

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load(nodes_only=True)

        assert exc.value.code == EXIT_MALFORMED_IDS

    def test_malformed_ids_outrank_validation_findings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent input is a stronger statement than a finding about what did land."""
        base = _summary_from_the_real_loader(tmp_path)
        finding = ValidationResult(
            query_name="chain_integrity",
            passed=False,
            details="100 cycle(s) detected",
            row_count=100,
        )
        summary = LoadSummary(
            node_results=base.node_results,
            edge_results=[],
            validation_results=[finding],
            total_nodes=base.total_nodes,
            validation_passed=False,
        )
        _install_load_stubs(monkeypatch, tmp_path, summary)

        with pytest.raises(SystemExit) as exc:
            cli._cmd_load()

        assert exc.value.code == EXIT_MALFORMED_IDS


def test_exit_code_does_not_collide() -> None:
    """`0` success, `2` argparse usage, `3` resolve's stop-at-limit.

    `1` and `4` are claimed by da#354. Pinned so a future code cannot silently
    reuse one.
    """
    assert EXIT_MALFORMED_IDS not in {0, 1, 2, 3, 4}
    assert EXIT_MALFORMED_IDS != EXIT_STOPPED_AT_LIMIT
