"""Application configuration via Pydantic Settings, loaded from .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j graph database connection settings."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "isnad_graph_dev"

    model_config = SettingsConfigDict(env_prefix="NEO4J_")


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    dsn: str = "postgresql://isnad:isnad_dev@localhost:5432/isnad_graph"

    model_config = SettingsConfigDict(env_prefix="PG_")


class Settings(BaseSettings):
    """Root application settings, composed from nested service settings."""

    neo4j: Neo4jSettings = Neo4jSettings()
    postgres: PostgresSettings = PostgresSettings()

    sunnah_api_key: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""

    data_raw_dir: Path = Path("./data/raw")
    data_staging_dir: Path = Path("./data/staging")
    data_curated_dir: Path = Path("./data/curated")

    topic_labels: list[str] = [
        "theology",
        "jurisprudence",
        "eschatology",
        "succession/imamate",
        "ritual/worship",
        "ethics/conduct",
        "history/sira",
        "commerce/trade",
        "warfare/jihad",
        "family_law",
        "food/drink",
        "medicine",
        "dreams/visions",
        "end_times",
    ]

    # Dedup embedding-encode parallelism (da#246). 0 = auto (scale to the box,
    # capped in dedup); 1 = serial; >1 = that many workers, clamped to cores.
    dedup_encode_workers: int = 0

    # Betweenness-centrality sampling pivots for the enrich metrics phase (da#326).
    # Exact Brandes betweenness is O(V*E) (~4e11 at 150k nodes / 2.68M edges) and
    # intractable at prod scale, so GDS uses sampled (approximate) Brandes with this
    # many pivot nodes — the choke-point *ranking* is preserved. 0 = exact (no
    # sampling); appropriate only for small graphs. Default 2000 pivots (~1.3% of the
    # 150k narrator graph). Deterministic across runs via betweenness_sampling_seed.
    betweenness_sampling_size: int = 2000
    betweenness_sampling_seed: int = 42

    log_level: str = "INFO"
    log_format: str = "console"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
