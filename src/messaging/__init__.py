"""Messaging helpers — Kafka producer for pipeline signal topics."""

from __future__ import annotations

from src.messaging.kafka_producer import (
    RAW_NEW_MESSAGE_SCHEMA,
    RawNewMessage,
    emit_raw_new,
    emit_raw_new_for_manifest,
)

__all__ = [
    "RAW_NEW_MESSAGE_SCHEMA",
    "RawNewMessage",
    "emit_raw_new",
    "emit_raw_new_for_manifest",
]
