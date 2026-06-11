# First-light vertical slice (da#73)

A **thin, vertical** path that gets real hadith data visible in the product
fast, ahead of the full multi-source pipeline (main#139). It takes the verified
keyless `sunnah_scraper` sample (riyadussalihin), runs it through parse and the
Phase-3 graph loader into the **staging Neo4j**, and confirms it is queryable
and visible in the isnad-graph frontend graph explorer.

This is deliberately scoped to **one collection**. It is not the full pipeline.

## What the sample is

`scripts/first_light/sample/sunnah_scraped/riyadussalihin.json` is a **real**
sample scraped live from sunnah.com (riyadussalihin, book 1) — 47 bilingual,
schema-valid hadiths. It is committed so the slice is reproducible offline and
in CI without network access. It is the same sample that da#71 verified.

## Counts (deterministic for this sample)

| Stage  | Result |
|--------|--------|
| Parse  | 47 hadith rows, 1 collection row, **0 source_id collisions** |
| Nodes  | 47 `Hadith` + 1 `Collection` |
| Edges  | 47 `APPEARS_IN` (hadith → collection), 0 missing endpoints |
| Grading | 0 (riyadussalihin grades are not exposed in the scraped HTML) |

These exact counts were loaded into the **staging Neo4j** (`noorinalabs-neo4j-1`,
v5.26.25) on 2026-06-10 — the documented verification query below returns
`riyadussalihin, 47` live, and the staging API `/health` reports `neo4j: up`.

## Run it

### Offline (acquire + parse only; no Neo4j)

```bash
uv run python scripts/first_light/run_slice.py --dry-run
```

Replays the committed sample, parses to staging Parquet, reports counts. Use
this to sanity-check the parse path anywhere.

### Load into the staging Neo4j

The staging Neo4j (`bolt://neo4j:7687`) is reachable only from **inside the
cluster** (or via a bolt tunnel) — it is not exposed publicly. Run from a box
that can resolve it, with the staging credentials in the environment:

```bash
export NEO4J_URI=bolt://neo4j:7687          # staging bolt endpoint
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<staging password>     # the deployed NEO4J_PASSWORD secret

uv run python scripts/first_light/run_slice.py            # replay committed sample
# or, to re-scrape live first:
uv run python scripts/first_light/run_slice.py --live --collection riyadussalihin --book 1
```

The loader is idempotent (`MERGE` on `id`), so re-running is safe.

> **Known loader bug surfaced by this slice (da#73, 2026-06-10).** The Phase-3
> edge loader's `_APPEARS_IN_QUERY` puts `hadith_number_in_book: row.hadith_number`
> **inside the relationship `MERGE` pattern**. Neo4j refuses to `MERGE` a
> relationship with a `null` property value:
>
> ```text
> Cannot merge the following relationship because of null property value for
> 'hadith_number_in_book': (h)-[:APPEARS_IN {hadith_number_in_book: null}]->(c)
> ```
>
> Because the scraper does not extract `hadith_number` (da#72), **every** scraped
> hadith has `hadith_number = null`, so the real `isnad-ingest load` aborts on the
> APPEARS_IN stage. The in-process mock test suite did not catch this (the mock
> counts batch rows; it does not enforce Neo4j's MERGE-null-property rule). The
> first-light staging load below was completed with a **null-safe** edge variant
> (`MERGE (h)-[r:APPEARS_IN]->(c) SET r.hadith_number_in_book = ...`). The loader
> fix is tracked separately (it touches the asserted MERGE contract in main#139
> and the edge-key assertions in #69/#74) and is **not** changed in this slice PR.

## Verify in Neo4j (Cypher)

The script prints and runs this after loading; you can also run it in Neo4j
Browser / `cypher-shell`:

```cypher
MATCH (c:Collection {id: 'col:sunnah:riyadussalihin'})
OPTIONAL MATCH (h:Hadith)-[:APPEARS_IN]->(c)
RETURN c.name_en AS collection,
       count(h) AS hadith_count,
       collect(h.matn_en)[0..3] AS sample_matn_en;
```

Expected: `collection = riyadussalihin`, `hadith_count = 47`, with real English
matn snippets in `sample_matn_en`.

## Verify in the frontend (graph explorer)

1. Open the staging frontend: <https://isnad-graph.noorinalabs.com> (graph
   explorer / search).
2. Search for `riyadussalihin` (or the collection node), or run the Cypher
   above in the explorer's query panel.
3. Confirm the `Collection` node with 47 `Hadith` nodes connected by
   `APPEARS_IN` renders. Capture a screenshot for the PR/issue evidence.

## Known caveat — da#72 (`hadith_number` not extracted)

`sunnah_scraper` does not currently extract `hadith_number`, so the
`source_id` (`sunnah:<collection>:<book>:<chapter>:<hadith_no|0>`) carries
`0` for the hadith-number segment. On **this** sample there are **no
collisions**, because each hadith in riyadussalihin book 1 sits in its own
synthetic chapter (101..147), keeping `source_id` distinct.

In collections where multiple hadiths share a chapter, the `0` hadith-number
would cause `source_id` collisions, and the loader's `MERGE` on `id` would
silently coalesce those hadiths into a single node. **da#72** (owned by
Alejandra Reyes-Fuentes) fixes the extraction; scaling this slice to
chapter-grouped collections should wait on it. The slice script prints a loud
`WARNING` if it ever sees a collision count > 0.
