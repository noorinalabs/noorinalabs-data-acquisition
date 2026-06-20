---
name: feedback_appears_in_merge_null
description: data-acquisition graph loader _APPEARS_IN_QUERY MERGEs the relationship with hadith_number_in_book inside the MERGE pattern → Neo4j rejects null property → real load aborts on every scraped (null hadith_number) hadith; mock suite masks it.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 080813cd-f3b8-434d-974c-badf58620c96
---

`src/graph/load_edges.py _APPEARS_IN_QUERY` builds the APPEARS_IN relationship
with `MERGE (h)-[:APPEARS_IN {book_number, chapter_number, hadith_number_in_book:
row.hadith_number}]->(c)` — positional props **inside** the MERGE pattern. Neo4j
**refuses to MERGE a relationship with a null property value**:
`Cannot merge ... null property value for 'hadith_number_in_book'`.

Because the sunnah scraper does not extract `hadith_number` (da#72), every
scraped hadith carries `hadith_number = null`, so the real `isnad-ingest load`
**aborts on the APPEARS_IN edge stage** for scraped data.

**Why:** the in-process `MockNeo4jClient` (`tests/test_graph/conftest.py`) only
counts batch rows in `execute_write_batch`; it never enforces Neo4j's
MERGE-null-property rule. So the full unit/mock suite is green while production
fails. Instance of [[feedback_test_mock_masks_prod_failure]] and
why live traces over synthetic acceptance matter — only the real
staging load (da#73, via SSH to [[project_staging_unreachable_from_sandbox]])
surfaced it.

**FIXED** da#77 branch `K.Boateng/0077-appears-in-null-safe-merge` (off wave-2):
`MERGE (h)-[r:APPEARS_IN]->(c) SET r.book_number = coalesce(row.book_number,
r.book_number), …, r.hadith_number_in_book = coalesce(row.hadith_number,
r.hadith_number_in_book)`. The (hadith, collection) PAIR is the edge identity, so
positional values are SET attributes, not MERGE-key — null-safe AND dedup-correct
(verified idempotent on live Neo4j: re-run creates 0). **Use coalesce-preserve,
not plain SET:** the streaming ingest path (`ingest-platform
workers/ingest/processor.py _build_edge_cypher`) already uses `r.<f> =
coalesce(row.<f>, r.<f>)` so an explicit-null preserves the existing value rather
than clearing it; matching it makes the batch + streaming paths converge
byte-for-byte on idempotency AND null-handling (verified live: set 5, null re-load
→ still 5). Plain `SET r.x = row.x` would CLEAR on null (Neo4j drops null-SET
props → key absent from `keys(r)`) — a real divergence from streaming, caught
during the main#139↔da#73 contract align by reading Nikolaos's processor. Coordinated as a contract change with Oyunbileg
(#69/#74 — unit string-contract test moves to the SET form; her read-back test on
a NON-null sample is unaffected), Nikolaos (ig#62/main#139 MERGE-shape harness),
Alejandra (da#72, orthogonal). The `AppearsIn` model is untouched.
Found 2026-06-10 da#73/PR#76, fixed 2026-06-11 da#77 (Kwesi).
