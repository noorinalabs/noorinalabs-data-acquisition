"""Pipeline topic constants — mirror of noorinalabs-isnad-ingest-platform/workers/lib/topics.py.

Must stay in sync with ``noorinalabs-isnad-ingest-platform/workers/lib/topics.py``
until a shared contracts package is extracted. Producers here and consumers
in the ingest-platform read from these literals; drift means dropped messages.
"""

from __future__ import annotations

PIPELINE_RAW_LANDED = "pipeline.raw.landed"
PIPELINE_DEDUP_DONE = "pipeline.dedup.done"
PIPELINE_ENRICH_DONE = "pipeline.enrich.done"
PIPELINE_NORMALIZE_DONE = "pipeline.normalize.done"
PIPELINE_DLQ = "pipeline.dlq"

__all__ = [
    "PIPELINE_DEDUP_DONE",
    "PIPELINE_DLQ",
    "PIPELINE_ENRICH_DONE",
    "PIPELINE_NORMALIZE_DONE",
    "PIPELINE_RAW_LANDED",
]
