"""Tests for the ``pipeline.raw.landed`` Kafka producer wrapper.

The producer emits the pipeline *pointer* contract (da#492) — a faithful
mirror of the consumer's ``PipelineMessage`` in
``noorinalabs-isnad-ingest-platform/workers/lib/message.py``. These tests
pin the five-field wire shape (``batch_id, source, b2_path, timestamp,
record_count``) and its ``extra="forbid"`` semantics so drift that would
crash the consumer's dedup worker is caught here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.messaging.kafka_producer import (
    DEFAULT_TOPIC,
    PIPELINE_MESSAGE_SCHEMA,
    PipelineMessage,
    emit_raw_new,
    emit_raw_new_for_manifest,
    serialize_message,
)

_CONTRACT_FIELDS = {"batch_id", "source", "b2_path", "timestamp", "record_count"}


class _FakeFuture:
    """Sentinel returned by ``_FakeProducer.send``; never awaited on the hot path."""


class _FakeProducer:
    """Mimics the kafka.KafkaProducer API the wrapper uses.

    The wrapper no longer calls ``future.get`` on emit (fire-and-forget);
    failures surface via ``send()`` raising, which mirrors kafka-python's
    behavior when the producer is closed or the buffer is full.
    """

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []
        self.fail_on_send = fail_on_send
        self.flushed = False
        self.closed = False

    def send(self, topic: str, *, value: bytes, key: bytes) -> _FakeFuture:
        if self.fail_on_send:
            raise RuntimeError("simulated kafka failure")
        self.sent.append((topic, value, key))
        return _FakeFuture()

    def flush(self, timeout: int) -> None:
        self.flushed = True

    def close(self, timeout: int) -> None:
        self.closed = True


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_id": "11111111-1111-4111-8111-111111111111",
        "source": "sunnah_api",
        "b2_path": "raw/sunnah_api/2026-04-21/c.json",
        "timestamp": "2026-04-21T00:00:00+00:00",
        "record_count": 0,
    }
    kwargs.update(overrides)
    return kwargs


class TestPipelineMessageContract:
    def test_serialize_emits_exactly_the_contract_fields(self) -> None:
        msg = PipelineMessage(**_valid_kwargs(record_count=7))
        decoded = json.loads(serialize_message(msg))
        assert set(decoded.keys()) == _CONTRACT_FIELDS
        assert decoded["batch_id"] == "11111111-1111-4111-8111-111111111111"
        assert decoded["source"] == "sunnah_api"
        assert decoded["b2_path"] == "raw/sunnah_api/2026-04-21/c.json"
        assert decoded["record_count"] == 7
        # timestamp round-trips to the same instant regardless of pydantic's
        # exact ISO rendering (Z vs +00:00).
        assert datetime.fromisoformat(decoded["timestamp"]) == datetime(2026, 4, 21, tzinfo=UTC)

    def test_serialize_is_utf8_bytes(self) -> None:
        raw = serialize_message(PipelineMessage(**_valid_kwargs()))
        assert isinstance(raw, bytes)
        raw.decode("utf-8")  # must not raise

    def test_round_trips_through_reparse(self) -> None:
        """serialize → parse identity — the property the consumer relies on."""
        msg = PipelineMessage(**_valid_kwargs(record_count=42))
        reparsed = PipelineMessage.model_validate(json.loads(serialize_message(msg)))
        assert reparsed == msg

    def test_extra_field_is_forbidden(self) -> None:
        """extra='forbid' — the consumer rejects unknown keys; so must the mirror."""
        with pytest.raises(ValidationError):
            PipelineMessage(**_valid_kwargs(checksum_sha256="a" * 64))

    def test_schema_constant_matches_model_fields(self) -> None:
        assert set(PIPELINE_MESSAGE_SCHEMA) == _CONTRACT_FIELDS
        assert set(PipelineMessage.model_fields) == _CONTRACT_FIELDS


class TestPipelineMessageValidation:
    def test_valid_payload_constructs(self) -> None:
        PipelineMessage(**_valid_kwargs())

    def test_negative_record_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            PipelineMessage(**_valid_kwargs(record_count=-1))

    def test_record_count_zero_is_allowed(self) -> None:
        assert PipelineMessage(**_valid_kwargs(record_count=0)).record_count == 0

    def test_non_iso_timestamp_raises(self) -> None:
        with pytest.raises(ValidationError):
            PipelineMessage(**_valid_kwargs(timestamp="yesterday"))

    def test_timestamp_default_is_utc_now(self) -> None:
        kwargs = _valid_kwargs()
        del kwargs["timestamp"]
        before = datetime.now(UTC)
        msg = PipelineMessage(**kwargs)
        assert msg.timestamp.tzinfo is not None
        assert before <= msg.timestamp <= datetime.now(UTC)

    def test_message_is_frozen(self) -> None:
        msg = PipelineMessage(**_valid_kwargs())
        with pytest.raises(ValidationError):
            msg.source = "other"  # type: ignore[misc]


class TestEmitRawNew:
    def test_noop_when_bootstrap_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        ok = emit_raw_new(source="sunnah_api", b2_path="raw/sunnah_api/2026-04-21/c.json")
        assert ok is True

    def test_sends_to_configured_topic_with_injected_producer(self) -> None:
        fake = _FakeProducer()
        ok = emit_raw_new(
            source="sunnah_api",
            b2_path="raw/sunnah_api/2026-04-21/c.json",
            record_count=10,
            batch_id="22222222-2222-4222-8222-222222222222",
            timestamp="2026-04-21T12:00:00+00:00",
            producer=fake,
        )
        assert ok is True
        assert len(fake.sent) == 1
        topic, value, key = fake.sent[0]
        assert topic == DEFAULT_TOPIC == "pipeline.raw.landed"
        assert key == b"raw/sunnah_api/2026-04-21/c.json"  # partition key is the object path
        payload = json.loads(value)
        assert set(payload.keys()) == _CONTRACT_FIELDS  # no stray fields reach the wire
        assert payload["source"] == "sunnah_api"
        assert payload["b2_path"] == "raw/sunnah_api/2026-04-21/c.json"
        assert payload["batch_id"] == "22222222-2222-4222-8222-222222222222"
        assert payload["record_count"] == 10
        assert datetime.fromisoformat(payload["timestamp"]) == datetime(2026, 4, 21, 12, tzinfo=UTC)

    def test_generates_batch_id_when_omitted(self) -> None:
        fake = _FakeProducer()
        emit_raw_new(source="lk_corpus", b2_path="raw/lk_corpus/2026-04-21/x.csv", producer=fake)
        import uuid

        batch_id = json.loads(fake.sent[0][1])["batch_id"]
        assert uuid.UUID(batch_id)  # a parseable UUID was minted

    def test_record_count_defaults_to_zero(self) -> None:
        fake = _FakeProducer()
        emit_raw_new(source="lk_corpus", b2_path="raw/lk_corpus/2026-04-21/x.csv", producer=fake)
        assert json.loads(fake.sent[0][1])["record_count"] == 0

    def test_default_topic_matches_canonical_constant(self) -> None:
        from src.messaging.topics import PIPELINE_RAW_LANDED

        assert DEFAULT_TOPIC == PIPELINE_RAW_LANDED == "pipeline.raw.landed"

    def test_custom_topic_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_RAW_LANDED_TOPIC", "alt.raw.landed")
        fake = _FakeProducer()
        emit_raw_new(source="lk_corpus", b2_path="raw/lk_corpus/2026-04-21/x.csv", producer=fake)
        assert fake.sent[0][0] == "alt.raw.landed"

    def test_returns_false_and_swallows_send_failure(self) -> None:
        fake = _FakeProducer(fail_on_send=True)
        ok = emit_raw_new(
            source="sunnah_api",
            b2_path="raw/sunnah_api/2026-04-21/c.json",
            producer=fake,
        )
        assert ok is False  # failure is surfaced, not raised

    def test_injected_producer_is_not_flushed_by_emit(self) -> None:
        """Caller owns flush/close for injected producers — fire-and-forget semantics."""
        fake = _FakeProducer()
        emit_raw_new(source="sunnah_api", b2_path="raw/sunnah_api/2026-04-21/c.json", producer=fake)
        assert fake.flushed is False
        assert fake.closed is False

    def test_builds_own_producer_and_closes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake) as build:
            ok = emit_raw_new(source="sunnah_api", b2_path="raw/sunnah_api/2026-04-21/c.json")
        assert ok is True
        assert build.call_count == 1
        assert fake.flushed is True
        assert fake.closed is True


class TestEmitRawNewForManifest:
    def _make_files(self, tmp_path: Path, names: list[str]) -> list[Path]:
        out = []
        for n in names:
            p = tmp_path / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"content-" + n.encode())
            out.append(p)
        return out

    def test_emits_one_per_file_with_expected_b2_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        files = self._make_files(tmp_path, ["a.json", "sub/b.csv"])

        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake):
            n = emit_raw_new_for_manifest(
                source="sunnah_api",
                local_dir=tmp_path,
                files=files,
                acquired_at="2026-04-21T00:00:00+00:00",
            )
        assert n == 2
        keys = [k.decode() for _, _, k in fake.sent]
        assert keys == [
            "raw/sunnah_api/2026-04-21/a.json",
            "raw/sunnah_api/2026-04-21/sub/b.csv",
        ]
        payload_a = json.loads(fake.sent[0][1])
        assert set(payload_a.keys()) == _CONTRACT_FIELDS
        assert payload_a["b2_path"] == "raw/sunnah_api/2026-04-21/a.json"
        assert payload_a["record_count"] == 0
        assert datetime.fromisoformat(payload_a["timestamp"]) == datetime(2026, 4, 21, tzinfo=UTC)

    def test_each_file_gets_a_distinct_batch_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        files = self._make_files(tmp_path, ["a.json", "b.json"])
        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake):
            emit_raw_new_for_manifest(source="sunnah_api", local_dir=tmp_path, files=files)
        batch_ids = {json.loads(v)["batch_id"] for _, v, _ in fake.sent}
        assert len(batch_ids) == 2  # each landed object is its own batch

    def test_missing_file_is_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        real = self._make_files(tmp_path, ["real.json"])
        phantom = tmp_path / "phantom.json"
        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake):
            n = emit_raw_new_for_manifest(
                source="fawaz",
                local_dir=tmp_path,
                files=[*real, phantom],
            )
        assert n == 1

    def test_empty_file_list_is_noop(self, tmp_path: Path) -> None:
        assert emit_raw_new_for_manifest(source="muhaddithat", local_dir=tmp_path, files=[]) == 0

    def test_per_file_failure_does_not_abort_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        files = self._make_files(tmp_path, ["a.json", "b.json", "c.json"])

        call_count = {"n": 0}

        class _FlakyProducer(_FakeProducer):
            def send(self, topic: str, *, value: bytes, key: bytes) -> _FakeFuture:
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("transient")
                return super().send(topic, value=value, key=key)

        flaky = _FlakyProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=flaky):
            n = emit_raw_new_for_manifest(source="open_hadith", local_dir=tmp_path, files=files)
        assert n == 2  # a and c succeeded, b failed

    def test_flush_happens_once_at_manifest_end_not_per_emit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The producer is flushed by the manifest helper on its way out —
        not by each per-file emit — so ``linger_ms`` batching is preserved."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        files = self._make_files(tmp_path, ["a.json", "b.json", "c.json"])

        flush_count = {"n": 0}

        class _CountingProducer(_FakeProducer):
            def flush(self, timeout: int) -> None:
                flush_count["n"] += 1
                super().flush(timeout)

        counting = _CountingProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=counting):
            n = emit_raw_new_for_manifest(source="sunnah_api", local_dir=tmp_path, files=files)
        assert n == 3
        # Exactly one flush at manifest end, NOT one per emit.
        assert flush_count["n"] == 1
        assert counting.closed is True

    @pytest.mark.parametrize(
        "iso_input",
        [
            "2026-04-22T03:27:33+00:00",
            "2026-04-22T03:27:33Z",
            "2026-04-22T03:27:33.123456+00:00",
            "2026-04-22T03:27:33-05:00",  # non-UTC offset, date in UTC-neutral local form
        ],
    )
    def test_b2_path_date_extracted_from_iso(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iso_input: str
    ) -> None:
        """acquired_at parsing must yield YYYY-MM-DD via fromisoformat, not slicing."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        files = self._make_files(tmp_path, ["a.json"])
        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake):
            n = emit_raw_new_for_manifest(
                source="sunnah_api",
                local_dir=tmp_path,
                files=files,
                acquired_at=iso_input,
            )
        assert n == 1
        key = fake.sent[0][2].decode()
        assert key.startswith("raw/sunnah_api/2026-04-22/")
