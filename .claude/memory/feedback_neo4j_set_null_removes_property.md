---
name: feedback_neo4j_set_null_removes_property
description: "Neo4j `SET r.prop = null` REMOVES the property (key absent), not sets it null; and `MERGE` aborts on a null inline-map property — the null-safe rewrite is property-less MERGE + SET, but null-valued rows then yield edges WITHOUT that key."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 080813cd-f3b8-434d-974c-badf58620c96
---

Two coupled Neo4j Cypher facts, verified against a real Neo4j 5.x (testcontainers) during da#77/da#69 coordination 2026-06-11:

1. `MERGE (h)-[:T {prop: row.x}]->(c)` **aborts** when `row.x` is null (Neo4j rejects null in a MERGE property map). The null-safe rewrite is `MERGE (h)-[r:T]->(c) SET r.prop = row.x`.

2. BUT `SET r.prop = null` **REMOVES the property** — it does not store a null-valued key. So under the SET-form rewrite, a null-valued row produces an edge that EXISTS (no abort — good) but whose `keys(r)` does NOT contain `prop`. Empirically: a 4-row batch with one null → `TOTAL_EDGES=4 WITH_CANONICAL_KEY=3`, null edge keys `['book_number','chapter_number']` (canonical key absent).

**Why:** this changes how you assert. A "null hadith_number → edge still created" regression test must assert edge EXISTENCE/count, NOT that the canonical key is present-with-null (it's absent). And a read-back test that asserts `"key" in keys(r)` for EVERY edge silently couples to the fixture being all-non-null — add a null row and it breaks under SET form.

**How to apply:** for null-safe MERGE rewrites, assert edge count for the null case; reserve `"key" in keys(r)` assertions for non-null fixtures and comment the coupling. Also: property-less `MERGE (h)-[r:T]->(c)` collapses multiple rows sharing the same (h,c) into ONE edge (distinguishing props moved to SET) — fine when h is unique per row, a merge-collision risk when not. Related: da#69 read-back pattern [[feedback_count_ge_zero_masks_empty_graph]].
