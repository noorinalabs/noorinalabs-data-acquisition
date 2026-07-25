"""Tests for src.enrich.metrics — graph metrics via Neo4j GDS."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from neo4j import exceptions as neo4j_exc

from src.enrich.metrics import _gds_available, _write_betweenness, run_metrics
from src.models.enrich import MetricsResult


@pytest.fixture
def mock_client() -> MagicMock:
    """A mock Neo4jClient with default read/write returns."""
    client = MagicMock()
    client.execute_read.return_value = []
    client.execute_write.return_value = []
    return client


def _betweenness_call(client: MagicMock):
    """Return the (args, kwargs) of the single gds.betweenness.write call."""
    calls = [c for c in client.execute_write.call_args_list if "gds.betweenness.write" in c.args[0]]
    assert len(calls) == 1, f"expected exactly one betweenness call, got {len(calls)}"
    return calls[0]


class TestGdsAvailable:
    """Tests for _gds_available helper."""

    def test_returns_true_when_gds_installed(self, mock_client: MagicMock) -> None:
        mock_client.execute_read.return_value = [{"version": "2.6.0"}]
        assert _gds_available(mock_client) is True
        mock_client.execute_read.assert_called_once_with("RETURN gds.version() AS version")

    def test_returns_false_when_gds_missing(self, mock_client: MagicMock) -> None:
        mock_client.execute_read.side_effect = neo4j_exc.Neo4jError(
            "Unknown function 'gds.version'"
        )
        assert _gds_available(mock_client) is False


class TestRunMetrics:
    """Tests for run_metrics orchestration."""

    def test_graceful_fallback_when_gds_unavailable(self, mock_client: MagicMock) -> None:
        mock_client.execute_read.side_effect = neo4j_exc.Neo4jError("gds not installed")
        result = run_metrics(mock_client)
        assert isinstance(result, MetricsResult)
        assert result.narrators_enriched == 0
        assert result.betweenness_computed is False
        assert result.pagerank_computed is False
        assert result.louvain_computed is False
        assert result.degree_computed is False
        assert result.communities_found == 0

    def test_graph_projection_call_sequence(self, mock_client: MagicMock) -> None:
        """Verify that run_metrics projects the graph, runs algos, and drops."""
        # _gds_available check
        read_responses = [
            [{"version": "2.6.0"}],  # gds.version()
            [{"exists": True}],  # gds.graph.exists
            [{"cnt": 42}],  # count enriched narrators
            [],  # top-5 query
        ]
        mock_client.execute_read.side_effect = read_responses
        mock_client.execute_write.return_value = [
            {"communityCount": 5, "nodePropertiesWritten": 10}
        ]

        result = run_metrics(mock_client)

        assert isinstance(result, MetricsResult)
        assert result.betweenness_computed is True
        assert result.pagerank_computed is True
        assert result.louvain_computed is True
        assert result.degree_computed is True
        assert result.communities_found == 5
        assert result.narrators_enriched == 42

        # Verify graph drop was called in the finally block
        write_calls = mock_client.execute_write.call_args_list
        last_write = write_calls[-1]
        assert "gds.graph.drop" in last_write.args[0]

    def test_returns_metrics_result(self, mock_client: MagicMock) -> None:
        mock_client.execute_read.side_effect = [
            [{"version": "2.6.0"}],  # gds check
            [{"exists": False}],  # graph exists
            [{"cnt": 10}],  # count
            [{"id": "n1", "name": "Test", "bc": 0.5}],  # top-5
        ]
        mock_client.execute_write.return_value = [
            {"communityCount": 3, "nodePropertiesWritten": 10}
        ]

        result = run_metrics(mock_client)
        assert isinstance(result, MetricsResult)
        data = result.model_dump()
        assert set(data.keys()) == {
            "narrators_enriched",
            "betweenness_computed",
            "pagerank_computed",
            "louvain_computed",
            "degree_computed",
            "communities_found",
        }


class TestBetweennessSampling:
    """da#326 — sampled vs exact betweenness tractability."""

    def _gds_up(self, mock_client: MagicMock) -> None:
        """Prime read side-effects so run_metrics runs the full algo sequence."""
        mock_client.execute_read.side_effect = [
            [{"version": "2.13.8"}],  # gds.version()
            [{"exists": False}],  # gds.graph.exists
            [{"cnt": 7}],  # count enriched
            [],  # top-5
        ]
        mock_client.execute_write.return_value = [{"communityCount": 2, "nodePropertiesWritten": 7}]

    def test_sampled_path_passes_sampling_params(self, mock_client: MagicMock) -> None:
        """A positive sampling_size runs sampled Brandes with size + seed."""
        self._gds_up(mock_client)

        run_metrics(mock_client, sampling_size=2000, sampling_seed=42)

        call = _betweenness_call(mock_client)
        query, params = call.args[0], call.args[1]
        assert "samplingSize: $samplingSize" in query
        assert "samplingSeed: $samplingSeed" in query
        assert params["samplingSize"] == 2000
        assert params["samplingSeed"] == 42

    def test_exact_path_when_sampling_size_none(self, mock_client: MagicMock) -> None:
        """Unset sampling_size (default) runs exact betweenness — no sampling params."""
        self._gds_up(mock_client)

        run_metrics(mock_client)  # sampling_size defaults to None

        call = _betweenness_call(mock_client)
        query, params = call.args[0], call.args[1]
        assert "samplingSize" not in query
        assert "samplingSeed" not in query
        assert params == {"name": "transmission_graph"}

    def test_exact_path_when_sampling_size_zero(self, mock_client: MagicMock) -> None:
        """sampling_size=0 is the explicit small-graph exact opt-out."""
        self._gds_up(mock_client)

        run_metrics(mock_client, sampling_size=0)

        call = _betweenness_call(mock_client)
        assert "samplingSize" not in call.args[0]

    def test_result_flags_set_on_sampled_run(self, mock_client: MagicMock) -> None:
        """betweenness_computed (and peers) are True on a successful sampled run."""
        self._gds_up(mock_client)

        result = run_metrics(mock_client, sampling_size=1500)

        assert result.betweenness_computed is True
        assert result.pagerank_computed is True
        assert result.louvain_computed is True
        assert result.degree_computed is True

    def test_write_betweenness_helper_sampled(self, mock_client: MagicMock) -> None:
        """_write_betweenness emits sampling params directly for a positive size."""
        _write_betweenness(mock_client, sampling_size=1000, sampling_seed=7)

        call = _betweenness_call(mock_client)
        assert call.args[1]["samplingSize"] == 1000
        assert call.args[1]["samplingSeed"] == 7

    def test_write_betweenness_helper_exact(self, mock_client: MagicMock) -> None:
        """_write_betweenness omits sampling params for None/0."""
        _write_betweenness(mock_client, sampling_size=None, sampling_seed=42)

        call = _betweenness_call(mock_client)
        assert "samplingSize" not in call.args[0]


def _leaderboard_query(client: MagicMock) -> str:
    """Return the top-5 betweenness leaderboard read query string."""
    calls = [
        c
        for c in client.execute_read.call_args_list
        if "ORDER BY bc" in c.args[0] and "LIMIT 5" in c.args[0]
    ]
    assert len(calls) == 1, f"expected exactly one top-5 leaderboard query, got {len(calls)}"
    return calls[0].args[0]


class TestLeaderboardName:
    """da#489 — top-5 leaderboard must project the always-populated name_en.

    The Narrator loader writes n.name_en (guaranteed non-empty via the
    transliteration fallback) and n.name_ar, but never n.name_arabic. Guard
    against a future rename silently reintroducing blank leaderboard names.
    """

    def _gds_up(self, mock_client: MagicMock) -> None:
        """Prime read side-effects so run_metrics reaches the top-5 query."""
        mock_client.execute_read.side_effect = [
            [{"version": "2.13.8"}],  # gds.version()
            [{"exists": False}],  # gds.graph.exists
            [{"cnt": 7}],  # count enriched
            [{"id": "nar:1", "name": "Test", "bc": 0.9}],  # top-5
        ]
        mock_client.execute_write.return_value = [{"communityCount": 2, "nodePropertiesWritten": 7}]

    def test_leaderboard_projects_name_en(self, mock_client: MagicMock) -> None:
        self._gds_up(mock_client)

        run_metrics(mock_client)

        query = _leaderboard_query(mock_client)
        assert "n.name_en AS name" in query

    def test_leaderboard_never_references_name_arabic(self, mock_client: MagicMock) -> None:
        self._gds_up(mock_client)

        run_metrics(mock_client)

        query = _leaderboard_query(mock_client)
        assert "name_arabic" not in query
