---
name: feedback_count_ge_zero_masks_empty_graph
description: "Integration assertion `count(r) >= 0` is vacuously true and masks a zero-edge/empty-graph bug; hardening to read-back the real graph surfaces it."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 080813cd-f3b8-434d-974c-badf58620c96
---

A real-DB integration test that closes with `MATCH ()-[r:TYPE]->() RETURN count(r) AS cnt` then `assert cnt >= 0` asserts NOTHING — it passes even when zero edges/rows were persisted, and regardless of property naming. In da#69 (data-acquisition PR #74) this masked a real fixture bug: `_load_appears_in` rebuilds the endpoint key as `col:{source_corpus}:{collection_name}` but `SAMPLE_COLLECTIONS` used plain ids (`col:bukhari` ≠ `col:sunnah:bukhari`), so 0 APPEARS_IN edges were ever created while the loader still returned a result object — `len(appears_in)==1` and `count>=0` both passed vacuously.

**Why:** the loader returns one `EdgeLoadResult` per edge TYPE regardless of `created`; counting result objects ≠ counting persisted edges.

**How to apply:** harden real-DB assertions to read back the actual graph — assert a concrete row/edge COUNT (cross-checked against the loader's `created`), then per-row assert the property `keys(r)` (canonical present + non-null, legacy absent). Also `pytest.skip` (not error) the container fixture on `docker.errors.DockerException` so a Docker-Hub/daemon outage degrades to SKIP, not a masking red ERROR. Related: [[feedback_test_mock_masks_prod_failure]].
