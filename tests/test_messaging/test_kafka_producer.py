"""Tests for the ``pipeline.raw.new`` Kafka producer wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.messaging.kafka_producer import (
    DEFAULT_TOPIC,
    RawNewMessage,
    _guess_content_type,
    emit_raw_new,
    emit_raw_new_for_manifest,
    sha256_of_file,
)


class _FakeFuture:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc

    def get(self, timeout: int) -> None:
        if self._exc is not None:
            raise self._exc


class _FakeProducer:
    """Mimics the kafka.KafkaProducer API the wrapper uses."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []
        self.fail_on_send = fail_on_send
        self.flushed = False
        self.closed = False

    def send(self, topic: str, *, value: bytes, key: bytes) -> _FakeFuture:
        if self.fail_on_send:
            return _FakeFuture(exc=RuntimeError("simulated kafka failure"))
        self.sent.append((topic, value, key))
        return _FakeFuture()

    def flush(self, timeout: int) -> None:
        self.flushed = True

    def close(self, timeout: int) -> None:
        self.closed = True


class TestRawNewMessage:
    def test_to_json_is_sorted_utf8(self) -> None:
        msg = RawNewMessage(
            source="sunnah_api",
            b2_key="raw/sunnah_api/2026-04-21/c.json",
            content_type="application/json",
            size_bytes=42,
            acquired_at="2026-04-21T00:00:00+00:00",
            checksum_sha256="deadbeef",
        )
        raw = msg.to_json()
        assert isinstance(raw, bytes)
        decoded = json.loads(raw)
        assert decoded == {
            "source": "sunnah_api",
            "b2_key": "raw/sunnah_api/2026-04-21/c.json",
            "content_type": "application/json",
            "size_bytes": 42,
            "acquired_at": "2026-04-21T00:00:00+00:00",
            "checksum_sha256": "deadbeef",
        }
        # keys sorted
        assert list(decoded.keys()) == sorted(decoded.keys())


class TestEmitRawNew:
    def test_noop_when_bootstrap_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        ok = emit_raw_new(
            source="sunnah_api",
            b2_key="raw/sunnah_api/2026-04-21/c.json",
            content_type="application/json",
            size_bytes=1,
            checksum_sha256="abc",
        )
        assert ok is True

    def test_sends_to_configured_topic_with_injected_producer(self) -> None:
        fake = _FakeProducer()
        ok = emit_raw_new(
            source="sunnah_api",
            b2_key="raw/sunnah_api/2026-04-21/c.json",
            content_type="application/json",
            size_bytes=10,
            checksum_sha256="cafe",
            acquired_at="2026-04-21T12:00:00+00:00",
            producer=fake,
        )
        assert ok is True
        assert len(fake.sent) == 1
        topic, value, key = fake.sent[0]
        assert topic == DEFAULT_TOPIC
        assert key == b"raw/sunnah_api/2026-04-21/c.json"
        payload = json.loads(value)
        assert payload["source"] == "sunnah_api"
        assert payload["size_bytes"] == 10
        assert payload["checksum_sha256"] == "cafe"
        assert payload["acquired_at"] == "2026-04-21T12:00:00+00:00"

    def test_custom_topic_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_RAW_NEW_TOPIC", "alt.raw.new")
        fake = _FakeProducer()
        emit_raw_new(
            source="lk_corpus",
            b2_key="raw/lk_corpus/2026-04-21/x.csv",
            content_type="text/csv",
            size_bytes=1,
            checksum_sha256="f",
            producer=fake,
        )
        assert fake.sent[0][0] == "alt.raw.new"

    def test_returns_false_and_swallows_send_failure(self) -> None:
        fake = _FakeProducer(fail_on_send=True)
        ok = emit_raw_new(
            source="sunnah_api",
            b2_key="raw/sunnah_api/2026-04-21/c.json",
            content_type="application/json",
            size_bytes=1,
            checksum_sha256="abc",
            producer=fake,
        )
        assert ok is False  # failure is surfaced, not raised

    def test_builds_own_producer_and_closes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        fake = _FakeProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=fake) as build:
            ok = emit_raw_new(
                source="sunnah_api",
                b2_key="raw/sunnah_api/2026-04-21/c.json",
                content_type="application/json",
                size_bytes=1,
                checksum_sha256="abc",
            )
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

    def test_emits_one_per_file_with_expected_b2_key(
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
        assert payload_a["content_type"] == "application/json"
        assert payload_a["size_bytes"] == len(b"content-a.json")

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
                    return _FakeFuture(exc=RuntimeError("transient"))
                return super().send(topic, value=value, key=key)

        flaky = _FlakyProducer()
        with patch("src.messaging.kafka_producer._build_producer", return_value=flaky):
            n = emit_raw_new_for_manifest(source="open_hadith", local_dir=tmp_path, files=files)
        assert n == 2  # a and c succeeded, b failed


class TestSha256AndContentType:
    def test_sha256_of_file_matches_hashlib(self, tmp_path: Path) -> None:
        import hashlib

        p = tmp_path / "x.bin"
        p.write_bytes(b"\x00\x01\x02")
        assert sha256_of_file(p) == hashlib.sha256(b"\x00\x01\x02").hexdigest()

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.json", "application/json"),
            ("a.csv", "text/csv"),
            ("a.PARQUET", "application/x-parquet"),
            ("a.unknown", "application/octet-stream"),
        ],
    )
    def test_guess_content_type(self, tmp_path: Path, name: str, expected: str) -> None:
        p = tmp_path / name
        p.touch()
        assert _guess_content_type(p) == expected


class TestBuildProducer:
    def test_returns_none_when_bootstrap_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        from src.messaging.kafka_producer import _build_producer

        assert _build_producer() is None

    def test_builds_kafka_producer_with_bootstrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker-a:9092, broker-b:9092")
        fake_kp_cls = MagicMock(return_value="producer-sentinel")
        fake_kafka_mod = MagicMock(KafkaProducer=fake_kp_cls)
        with patch.dict("sys.modules", {"kafka": fake_kafka_mod}):
            from src.messaging.kafka_producer import _build_producer

            result = _build_producer()
        assert result == "producer-sentinel"
        call_kwargs: dict[str, Any] = fake_kp_cls.call_args.kwargs
        assert call_kwargs["bootstrap_servers"] == ["broker-a:9092", "broker-b:9092"]
        assert call_kwargs["acks"] == "all"
        assert call_kwargs["retries"] == 3
