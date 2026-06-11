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

The loader is idempotent (`MERGE` on `id`), so re-running is safe. The
APPEARS_IN edge loader's null-safety (a real bug this slice surfaced on
2026-06-10) is tracked as **da#77** — see the loader note under "History" below.

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

## Visual evidence — live run confirmed

`scripts/first_light/evidence/firstlight_graph.png` is the loaded graph rendered
directly from a **live Neo4j 5**: the `riyadussalihin` Collection node (red) with
its 47 `Hadith` nodes (green) connected by `APPEARS_IN`, each Hadith labelled
with its in-book `hadith_number` (1..47). Regenerate it with:

```bash
uv run --with matplotlib --with networkx python scripts/first_light/render_graph.py
```

This run was through the **real loader** (`run_slice.py` → `load_all`), end to
end. With the da#72 scraper fix merged (da#75), `hadith_number` is the genuine
**in-book ordinal** (1, 2, 3 … within the book), so the `APPEARS_IN`
`hadith_number_in_book` edge property carries the semantically-correct value.

## Verify in the frontend (graph explorer)

1. Open the staging frontend: <https://isnad-graph.noorinalabs.com> (graph
   explorer / search).
2. Search for `riyadussalihin` (or the collection node), or run the Cypher
   above in the explorer's query panel.
3. Confirm the `Collection` node with 47 `Hadith` nodes connected by
   `APPEARS_IN` renders. Capture a screenshot for the PR/issue evidence.
   (The product graph-explorer view requires an authenticated session; the PNG
   above is the equivalent direct-from-Neo4j render.)

## History — `hadith_number` keying (da#72 / da#75, resolved)

Earlier revisions of this slice ran before the scraper extracted `hadith_number`:
`source_id` carried `0` for the hadith-number segment
(`sunnah:<collection>:<book>:<chapter>:0`), kept collision-free on this sample
only because each riyadussalihin book-1 hadith sat in its own synthetic chapter.

**da#72 (merged as da#75)** fixed this: the scraper now extracts the genuine
**in-book ordinal** (the "Book N, Hadith M" reference) into `hadith_number`, so
`source_id` is `sunnah:<collection>:<book>:<chapter>:<in-book-ordinal>` —
collision-free on its own keying (47/47 distinct), and the
`hadith_number_in_book` edge property now matches its name. The collection-wide
reference number (e.g. "Riyad as-Salihin 680") is intentionally kept only as a
human label, not in `source_id`. The slice script still prints a loud `WARNING`
if it ever sees a `source_id` collision.

> **Loader note (da#77).** The Phase-3 edge loader's `_APPEARS_IN_QUERY`
> historically put `hadith_number_in_book` **inside** the relationship `MERGE`
> pattern, which Neo4j rejects on a null value — so any null `hadith_number`
> aborted the load. da#77 makes the loader null-safe (`MERGE` on the
> `(hadith, collection)` pair, then `SET … = coalesce(…)`). With da#75 merged the
> scraped value is non-null, but the null-safe loader remains the correct
> contract for any source that leaves the field unset.
