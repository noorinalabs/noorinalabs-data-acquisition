"""Messaging helpers — Kafka producer for pipeline signal topics."""

from __future__ import annotations

from src.messaging.kafka_producer import (
    PIPELINE_MESSAGE_SCHEMA,
    PipelineMessage,
    emit_raw_new,
    emit_raw_new_for_manifest,
    serialize_message,
)

__all__ = [
    "PIPELINE_MESSAGE_SCHEMA",
    "PipelineMessage",
    "emit_raw_new",
    "emit_raw_new_for_manifest",
    "serialize_message",
]
