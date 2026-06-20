---
name: project_studied_under_allowlist
description: STUDIED_UNDER loader globs network_edges_*.parquet but only studentship sources (allowlist); isnad-transmission producers must be excluded
metadata: 
  node_type: memory
  type: project
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

da-ingestion `src/graph/load_edges.py` `_load_studied_under` globs ALL `network_edges_*.parquet` (since #130) but loads them as STUDIED_UNDER (teacher↔student). NETWORK_EDGE_SCHEMA is reused by producers meaning DIFFERENT relations: muhaddithat + itqan = studentship (correct), but `mis` (da#97) rows are ISNAD TRANSMISSION (different type AND opposite direction). Globbing mis in → wrong-type/wrong-direction edges.

Interim fix (da#97, PR #132): `_STUDIED_UNDER_SOURCES = {"muhaddithat","itqan"}` allowlist + `_is_studied_under_file` (exact slug or chunked `<slug>_NNN`). **Any new adapter that emits `network_edges_<slug>.parquet` whose edges are NOT studentship must stay OFF this allowlist.** Durable fix = explicit edge-relation field on NETWORK_EDGE_SCHEMA so the loader keys on data not slug — tracked in **da#133**. See [[feedback_security_guard_inline_not_followup]] (added risk + mitigation in same PR).
