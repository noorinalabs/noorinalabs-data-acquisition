"""Kafka producer for the ``pipeline.raw.landed`` signal topic.

After an acquire connector lands a raw artifact in B2, emit one
``pipeline.raw.landed`` message per file so downstream pipeline workers
(dedup, enrich, normalize, graph-load) can consume. The topic string
is defined in :mod:`src.messaging.topics` as ``PIPELINE_RAW_LANDED``.

Message schema — the pipeline pointer contract (da#492)
-------------------------------------------------------

The wire message is the pipeline *pointer* schema, aligned to the
consumer's ``PipelineMessage`` in
``noorinalabs-isnad-ingest-platform/workers/lib/message.py`` (contract
issue #106). Kafka messages are lightweight pointers — the raw bytes
live in S3-compatible storage (B2 in prod, MinIO for local dev) and the
message carries a ``b2_path`` that identifies the object.

::

    {
      "batch_id":     "uuid",                                   // this processing batch
      "source":       "sunnah_api",                             // connector name
      "b2_path":      "raw/sunnah_api/2026-04-21/collections.json",
      "timestamp":    "2026-04-21T12:34:56.789012+00:00",       // ISO-8601 UTC
      "record_count": 0                                         // records in the object
    }

The consumer validates with ``extra="forbid"`` — the producer MUST emit
exactly these five fields and no others. :class:`PipelineMessage` below
is a faithful mirror of the consumer model; it must stay in sync with
``workers/lib/message.py`` until a shared contracts package is extracted
(audit A4). Drift here silently drops every message on the floor at the
consumer's validation boundary — the failure mode da#492 fixed.

``record_count`` at the raw-landed stage is the count of records in the
referenced object. The acquire stage lands raw bytes without parsing
them, so the count is not known here; it is emitted as ``0`` (the
contract permits ``>= 0``) and the dedup/parse stage establishes the
true count when it reads the object and hands off downstream.

The message is a purely "eventually consistent" signal — emit failure
MUST NOT fail the acquire. The downstream dedup worker is responsible
for handling duplicate messages; re-acquisition of the same B2 key is
allowed to re-emit and is intentionally not deduplicated here
(:issue:`28`).

Configuration (env)
-------------------
``KAFKA_BOOTSTRAP_SERVERS``    — comma-separated host:port list
                                 (e.g. ``kafka:9092`` locally,
                                  ``broker-1:9092,broker-2:9092`` in prod)
``KAFKA_RAW_LANDED_TOPIC``     — topic name (default ``pipeline.raw.landed``)

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

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.messaging.topics import PIPELINE_RAW_LANDED
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_TOPIC",
    "PIPELINE_MESSAGE_SCHEMA",
    "PipelineMessage",
    "emit_raw_new",
    "emit_raw_new_for_manifest",
    "serialize_message",
]

DEFAULT_TOPIC = PIPELINE_RAW_LANDED

PIPELINE_MESSAGE_SCHEMA: dict[str, str] = {
    "batch_id": "UUID identifying this processing batch",
    "source": "connector name (e.g. sunnah_api)",
    "b2_path": "full B2 object key, e.g. raw/sunnah_api/2026-04-21/collections.json",
    "timestamp": "ISO-8601 UTC timestamp of production",
    "record_count": "number of records in the referenced object (int >= 0)",
}


class PipelineMessage(BaseModel):
    """Pointer message flowing between pipeline stages (producer side).

    Faithful mirror of the consumer's ``PipelineMessage`` in
    ``noorinalabs-isnad-ingest-platform/workers/lib/message.py`` (contract
    issue #106). Field names, types and ``extra="forbid"`` must match the
    consumer exactly — see the module docstring. Serialize with
    :func:`serialize_message` for the wire.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str = Field(..., description="UUID identifying this processing batch")
    source: str = Field(..., description="Data source identifier, e.g. 'sunnah_api'")
    b2_path: str = Field(..., description="S3 object key or folder prefix of the raw artifact")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this message was produced (ISO-8601 UTC)",
    )
    record_count: int = Field(..., ge=0, description="Number of records in the referenced object")


def serialize_message(msg: PipelineMessage) -> bytes:
    """Serialize a :class:`PipelineMessage` to Kafka wire bytes (UTF-8 JSON).

    Mirrors the consumer's ``serialize_message`` so a producer→consumer
    round-trip (``parse_message(serialize_message(msg))``) is a byte-faithful
    identity. ``pydantic`` renders ``timestamp`` as an ISO-8601 string and
    emits exactly the five contract fields.
    """
    return msg.model_dump_json().encode("utf-8")


def _bootstrap_servers() -> str:
    """Return configured brokers or empty string when unset."""
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()


def _topic() -> str:
    return os.environ.get("KAFKA_RAW_LANDED_TOPIC", DEFAULT_TOPIC).strip() or DEFAULT_TOPIC


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
    b2_path: str,
    record_count: int = 0,
    batch_id: str | None = None,
    timestamp: str | datetime | None = None,
    producer: Any | None = None,  # noqa: ANN401
) -> bool:
    """Emit a single ``pipeline.raw.landed`` message.

    Emit is fire-and-forget — ``send()`` returns without blocking on the
    broker ACK. Callers that inject their own ``producer`` are responsible
    for calling ``producer.flush()`` to guarantee delivery. When this
    function owns the producer (no injection, broker configured), it
    flushes before returning so durability holds for the one-off path.

    Returns ``True`` on successful enqueue (or producer no-op when
    ``KAFKA_BOOTSTRAP_SERVERS`` is unset), ``False`` when the send call
    itself raised. Never raises on send failure — failures are logged and
    swallowed (see module docstring § Retry semantics). A malformed
    payload (caught at :class:`PipelineMessage` construction) DOES raise:
    an ill-formed message is a programming error, not a transient fault.

    Parameters
    ----------
    source, b2_path, record_count
        See :data:`PIPELINE_MESSAGE_SCHEMA`. ``record_count`` defaults to
        ``0`` — the raw-landed stage does not parse the object, so the true
        count is established downstream.
    batch_id
        UUID for this batch. Defaults to a fresh ``uuid4`` — each landed
        object is an independently processed batch flowing 1:1 through the
        pipeline stages.
    timestamp
        Production time. ISO-8601 string or ``datetime``; defaults to now
        (UTC).
    producer
        Injected producer — used by tests and by
        :func:`emit_raw_new_for_manifest` to share a producer across
        many per-file emits. Production one-off callers omit this.
    """
    # ``model_validate`` (not the typed kwargs constructor) so pydantic is the
    # single validator/coercer for every field — including ``timestamp``, which
    # callers may pass as an ISO-8601 string or a ``datetime``.
    msg = PipelineMessage.model_validate(
        {
            "batch_id": batch_id if batch_id is not None else str(uuid.uuid4()),
            "source": source,
            "b2_path": b2_path,
            "timestamp": timestamp if timestamp is not None else datetime.now(UTC),
            "record_count": record_count,
        }
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
                b2_path=b2_path,
            )
            return True

    try:
        producer.send(topic, value=serialize_message(msg), key=b2_path.encode("utf-8"))
        logger.info(
            "kafka_emit_enqueued",
            topic=topic,
            source=source,
            b2_path=b2_path,
            batch_id=msg.batch_id,
            record_count=record_count,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — must never propagate
        logger.error(
            "kafka_emit_failed",
            topic=topic,
            source=source,
            b2_path=b2_path,
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
    """Emit one ``pipeline.raw.landed`` message per file.

    Computes the ``b2_path`` as ``<b2_prefix>/<relative-path>`` where
    ``b2_prefix`` defaults to ``raw/<source>/<YYYY-MM-DD>/``. Each file is
    emitted as its own batch (fresh ``batch_id``) with ``record_count=0``
    — the acquire stage lands raw bytes without parsing them, so the count
    is established downstream when the object is read.

    ``acquired_at`` is the acquisition time; it is mapped to the wire
    ``timestamp`` field and also used to derive the ``YYYY-MM-DD`` date
    segment of the default ``b2_prefix``.

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
            b2_path = f"{prefix}/{rel.as_posix()}"
            if emit_raw_new(
                source=source,
                b2_path=b2_path,
                timestamp=ts,
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
