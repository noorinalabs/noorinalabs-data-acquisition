"""Integration test fixtures using testcontainers.

These tests REQUIRE a working Docker daemon: testcontainers spins up real
Neo4j / Postgres containers. When Docker is unreachable — e.g. a dev box with
no daemon, or a CI runner that cannot pull the testcontainers reaper image from
Docker Hub — the ``neo4j_container`` fixture ``pytest.skip``s the leg rather
than erroring, so a missing-infra environment yields a clean SKIP instead of a
red ERROR that masks real signal (#69).
"""

from __future__ import annotations

import docker.errors
import pytest
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from src.utils.neo4j_client import Neo4jClient

NEO4J_TEST_PASSWORD = "testpassword123"
# Single source of truth for the testcontainers Neo4j base image tag.
# Org-standardized on neo4j:5-community (noorinalabs-main#642) to share the image
# cache and avoid redundant CI pulls. This is the upstream community base image,
# NOT the app's custom isnad-graph-neo4j:5-community (which bundles APOC/plugins).
NEO4J_TEST_IMAGE = "neo4j:5-community"


@pytest.fixture(scope="session")
def neo4j_container():
    """Start a real Neo4j container for integration tests.

    Requires Docker. Skips (does not error) when Docker is unavailable so the
    suite degrades gracefully off-CI / on a runner that cannot reach Docker Hub.
    """
    container = Neo4jContainer(NEO4J_TEST_IMAGE, password=NEO4J_TEST_PASSWORD)
    try:
        # ``start`` is where Docker is actually contacted (daemon connect, image
        # pull, container create); guard only this so a real test-body failure
        # later is never mistaken for an infra skip.
        neo4j = container.start()
    except docker.errors.DockerException as exc:
        pytest.skip(f"Docker/Neo4j unavailable — skipping integration tests: {exc}")
    try:
        yield neo4j
    finally:
        container.stop()


@pytest.fixture
def neo4j_client(neo4j_container):
    """Neo4jClient connected to the test container."""
    client = Neo4jClient(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=NEO4J_TEST_PASSWORD,
    )
    yield client
    # Clean up: delete all nodes after each test
    client.execute_write("MATCH (n) DETACH DELETE n")
    client.close()


@pytest.fixture(scope="session")
def postgres_container():
    """Start a real PostgreSQL container."""
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg


@pytest.fixture
def sample_data_dir(tmp_path):
    """Path to a temporary sample data directory for integration tests."""
    sample_dir = tmp_path / "test_samples"
    sample_dir.mkdir()
    return sample_dir


@pytest.fixture
def clean_staging(tmp_path):
    """Cleanup fixture that empties staging dir between tests."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(exist_ok=True)
    yield staging_dir
    # Cleanup after test
    for f in staging_dir.iterdir():
        if f.is_file():
            f.unlink()
