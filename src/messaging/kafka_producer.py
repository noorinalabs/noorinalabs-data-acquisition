"""Kafka producer for the ``pipeline.raw.new`` signal topic.

After an acquire connector lands a raw artifact in B2, emit one
``pipeline.raw.new`` message per file so downstream pipeline workers
(dedup, enrich, normalize, graph-load) can consume.

Message schema (JSON, UTF-8, one message per file)
---------------------------------------------------

::

    {
      "source":         "<connector_name>",          // e.g. sunnah_api, lk_corpus
      "b2_key":         "raw/<source>/<date>/<filename>",
      "content_type":   "application/json|text/csv|...",
      "size_bytes":     12345,
      "acquired_at":    "2026-04-21T12:34:56.789012+00:00",  // ISO-8601 UTC
      "checksum_sha256": "<hex>"
    }

The message is a purely "eventually consistent" signal — emit failure
MUST NOT fail the acquire. The downstream dedup worker is responsible
for handling duplicate messages; re-acquisition of the same B2 key is
allowed to re-emit and is intentionally not deduplicated here
(:issue:`28`).

Configuration (env)
-------------------
``KAFKA_BOOTSTRAP_SERVERS``   — comma-separated host:port list
                                (e.g. ``kafka:9092`` locally,
                                 ``broker-1:9092,broker-2:9092`` in prod)
``KAFKA_RAW_NEW_TOPIC``       — topic name (default ``pipeline.raw.new``)

If ``KAFKA_BOOTSTRAP_SERVERS`` is unset/empty the producer is a no-op —
messages are logged at debug level and dropped. This keeps the acquire
stage runnable in dev/test environments without a running Kafka broker.

Retry semantics
---------------
``kafka-python`` applies its own bounded in-process retry. We configure
``retries=3`` and ``retry_backoff_ms=500``. The wrapper does NOT add
client-side jitter — if multiple connectors recover simultaneously they
may briefly stampede the broker. This is acceptable at current scale
(8 connectors, bounded throughput); revisit if connector count grows.

Delivery semantics
------------------
Emit is fire-and-forget: ``send()`` returns without blocking on the
broker ACK. Durability is guaranteed only by calling ``producer.flush()``
before process exit. :func:`emit_raw_new_for_manifest` flushes the
owned producer on its way out; callers that inject their own producer
(or invoke :func:`emit_raw_new` one-off) must flush themselves to
guarantee delivery.

Rationale: a per-emit ``future.get()`` would serialize sends and defeat
``linger_ms`` batching — 1k files would become 1k serial round-trips.
B2 is the source of truth; a missed Kafka emit can be reconciled by
replaying keys from B2.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_TOPIC",
    "RAW_NEW_MESSAGE_SCHEMA",
    "RawNewMessage",
    "emit_raw_new",
    "emit_raw_new_for_manifest",
    "sha256_of_file",
]

DEFAULT_TOPIC = "pipeline.raw.new"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RAW_NEW_MESSAGE_SCHEMA: dict[str, str] = {
    "source": "connector name (e.g. sunnah_api)",
    "b2_key": "full B2 object key, e.g. raw/sunnah_api/2026-04-21/collections.json",
    "content_type": "MIME type of the raw object",
    "size_bytes": "object size in bytes (int)",
    "acquired_at": "ISO-8601 UTC timestamp of acquisition",
    "checksum_sha256": "SHA-256 hex digest of raw bytes",
}


@dataclass(frozen=True, slots=True)
class RawNewMessage:
    """Typed payload for the ``pipeline.raw.new`` topic."""

    source: str
    b2_key: str
    content_type: str
    size_bytes: int
    acquired_at: str
    checksum_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.b2_key, str) or not self.b2_key:
            raise ValueError("b2_key must be a non-empty string")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise ValueError("size_bytes must be an int")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
        if not isinstance(self.checksum_sha256, str) or not _SHA256_RE.match(self.checksum_sha256):
            raise ValueError("checksum_sha256 must be a 64-char lowercase hex SHA-256 digest")
        if not isinstance(self.acquired_at, str):
            raise ValueError("acquired_at must be an ISO-8601 string")
        try:
            datetime.fromisoformat(self.acquired_at)
        except ValueError as exc:
            raise ValueError(f"acquired_at must be ISO-8601 parseable: {exc}") from exc

    def to_json(self) -> bytes:
        """Serialize to canonical JSON bytes (sorted keys, UTF-8)."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _bootstrap_servers() -> str:
    """Return configured brokers or empty string when unset."""
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()


def _topic() -> str:
    return os.environ.get("KAFKA_RAW_NEW_TOPIC", DEFAULT_TOPIC).strip() or DEFAULT_TOPIC


def _build_producer() -> Any:  # noqa: ANN401 — KafkaProducer has no public stub
    """Construct a ``kafka.KafkaProducer`` or return ``None`` when unconfigured.

    Import is deferred so test runs that never emit never import ``kafka``.
    """
    servers = _bootstrap_servers()
    if not servers:
        return None

    from kafka import KafkaProducer  # noqa: PLC0415 — deferred import

    return KafkaProducer(
        bootstrap_servers=[s.strip() for s in servers.split(",") if s.strip()],
        value_serializer=lambda v: v,  # we pass bytes already
        acks="all",
        retries=3,
        retry_backoff_ms=500,
        linger_ms=100,
        request_timeout_ms=10_000,
        max_block_ms=5_000,
    )


def emit_raw_new(
    *,
    source: str,
    b2_key: str,
    content_type: str,
    size_bytes: int,
    checksum_sha256: str,
    acquired_at: str | None = None,
    producer: Any | None = None,  # noqa: ANN401
) -> bool:
    """Emit a single ``pipeline.raw.new`` message.

    Emit is fire-and-forget — ``send()`` returns without blocking on the
    broker ACK. Callers that inject their own ``producer`` are responsible
    for calling ``producer.flush()`` to guarantee delivery. When this
    function owns the producer (no injection, broker configured), it
    flushes before returning so durability holds for the one-off path.

    Returns ``True`` on successful enqueue (or producer no-op when
    ``KAFKA_BOOTSTRAP_SERVERS`` is unset), ``False`` when the send call
    itself raised. Never raises — failures are logged and swallowed
    (see module docstring § Retry semantics).

    Parameters
    ----------
    source, b2_key, content_type, size_bytes, checksum_sha256
        See :data:`RAW_NEW_MESSAGE_SCHEMA`.
    acquired_at
        ISO-8601 timestamp. Defaults to now (UTC).
    producer
        Injected producer — used by tests and by
        :func:`emit_raw_new_for_manifest` to share a producer across
        many per-file emits. Production one-off callers omit this.
    """
    msg = RawNewMessage(
        source=source,
        b2_key=b2_key,
        content_type=content_type,
        size_bytes=size_bytes,
        acquired_at=acquired_at or datetime.now(UTC).isoformat(),
        checksum_sha256=checksum_sha256,
    )

    topic = _topic()
    owns_producer = False
    if producer is None:
        producer = _build_producer()
        owns_producer = True
        if producer is None:
            logger.debug(
                "kafka_emit_skipped",
                reason="KAFKA_BOOTSTRAP_SERVERS unset",
                topic=topic,
                b2_key=b2_key,
            )
            return True

    try:
        producer.send(topic, value=msg.to_json(), key=b2_key.encode("utf-8"))
        logger.info(
            "kafka_emit_enqueued",
            topic=topic,
            source=source,
            b2_key=b2_key,
            size_bytes=size_bytes,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — must never propagate
        logger.error(
            "kafka_emit_failed",
            topic=topic,
            source=source,
            b2_key=b2_key,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False
    finally:
        if owns_producer:
            try:
                producer.flush(timeout=10)
                producer.close(timeout=5)
            except Exception:  # noqa: BLE001
                pass


def emit_raw_new_for_manifest(
    *,
    source: str,
    local_dir: Path,
    files: list[Path],
    b2_prefix: str | None = None,
    acquired_at: str | None = None,
) -> int:
    """Emit one ``pipeline.raw.new`` message per file.

    Computes the ``b2_key`` as ``<b2_prefix>/<relative-path>`` where
    ``b2_prefix`` defaults to ``raw/<source>/<YYYY-MM-DD>/``. ``size_bytes``
    and ``checksum_sha256`` are read from the local file on disk — the
    acquire stage writes locally before the pipeline's B2 upload shim
    mirrors to the bucket, so the local file is the authoritative
    artifact.

    Emission failures are logged per-file and do NOT abort the batch.
    Returns the count of messages successfully sent.
    """
    if not files:
        return 0

    if acquired_at is None:
        date_utc = datetime.now(UTC).strftime("%Y-%m-%d")
    else:
        # Parse via fromisoformat (handles `Z` suffix on 3.11+) to extract
        # a guaranteed YYYY-MM-DD date — slicing `acquired_at[:10]` would
        # silently mis-slice non-ISO inputs.
        iso = acquired_at.replace("Z", "+00:00") if acquired_at.endswith("Z") else acquired_at
        date_utc = datetime.fromisoformat(iso).date().isoformat()
    prefix = (b2_prefix or f"raw/{source}/{date_utc}").rstrip("/")

    producer = _build_producer()
    owns_producer = producer is not None
    try:
        sent = 0
        ts = acquired_at or datetime.now(UTC).isoformat()
        for f in files:
            if not f.is_file():
                logger.warning("kafka_emit_skipped_missing_file", path=str(f), source=source)
                continue
            rel = f.relative_to(local_dir) if f.is_relative_to(local_dir) else Path(f.name)
            b2_key = f"{prefix}/{rel.as_posix()}"
            if emit_raw_new(
                source=source,
                b2_key=b2_key,
                content_type=_guess_content_type(f),
                size_bytes=f.stat().st_size,
                checksum_sha256=sha256_of_file(f),
                acquired_at=ts,
                producer=producer,
            ):
                sent += 1
        return sent
    finally:
        if owns_producer and producer is not None:
            try:
                producer.flush(timeout=10)
                producer.close(timeout=5)
            except Exception:  # noqa: BLE001
                pass


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_EXT_TO_CT = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".parquet": "application/x-parquet",
    ".xml": "application/xml",
    ".html": "text/html",
    ".txt": "text/plain",
    ".zip": "application/zip",
    ".gz": "application/gzip",
}


def _guess_content_type(path: Path) -> str:
    return _EXT_TO_CT.get(path.suffix.lower(), "application/octet-stream")
