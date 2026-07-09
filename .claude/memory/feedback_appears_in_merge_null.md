---
name: feedback_appears_in_merge_null
description: "HISTORICAL (fixed da#77): _APPEARS_IN_QUERY once MERGEd hadith_number_in_book inside the MERGE pattern → Neo4j rejected the null property → real load aborted on every scraped hadith; mock suite masked it. At HEAD the query is the coalesce-after-MERGE form and is null-safe — this memory explains WHY that shape exists, it does NOT describe a live defect."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 080813cd-f3b8-434d-974c-badf58620c96
---

> **STALE-AT-HEAD as of 2026-07-09 (verified during da#353 / PR #357).** The
> defect described below is **fixed** and has been since da#77. `src/graph/
> load_edges.py:440-448` at HEAD is the null-safe coalesce-after-MERGE form:
> `MERGE (h)-[r:APPEARS_IN]->(c)` with **no properties in the pattern**, then
> `SET r.hadith_number_in_book = coalesce(row.hadith_number, r.hadith_number_in_book)`.
> Do **not** read this memory as a live constraint — in particular, **emitting a
> null `book_number` or `hadith_number` from a parser is safe**. This file is
> retained because the historical failure is real and records *why* the current
> shape exists; the reasoning below (MERGE-key vs SET attribute, coalesce-preserve
> vs plain SET) is still binding. Re-verify against `load_edges.py` before citing.

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

**2026-07-09 (da#353/PR#357):** re-verified at HEAD by two reviewers + the TPM —
`load_edges.py:440-448` carries the coalesce-after-MERGE form; the pattern holds
no properties. The stale present-tense summary on this memory nearly deterred a
correct parser change that emits a null `book_number`. Lesson beyond the Cypher:
**a memory that records a fixed bug must say so in its `description`,** because
the description is what `MEMORY.md` and recall surface — a body-buried "FIXED"
line does not reach the reader in time. See [[feedback_full_read_over_tail]].
