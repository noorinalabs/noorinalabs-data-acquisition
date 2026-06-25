# ADR-003: `sanadset` orphan-node + narrator-pollution remediation

## Status: Proposed — awaiting owner A/B decision (P7W19, da#202)

> This ADR is an **investigation + A/B recommendation**, not an accepted
> decision. The remediation it scopes (a per-source purge) is irreversible on
> live graph state and is **owner-run only**. Nothing here has been executed.
> Meta-issue: noorinalabs/noorinalabs-main#723. Keystone: da#202.

## Context

Production validation of the live graph (Admin -> Data Management, 2026-06-19)
found that ~85% of all `Hadith` nodes are orphans — present, but linked to no
`Collection`. The provenance breakdown by `source_corpus`:

| source        | hadiths | collections |
|---------------|--------:|------------:|
| `sanadset`    | 650,986 |           0 |
| `lk`          |  33,981 |           6 |
| `thaqalayn`   |  33,190 |          33 |
| `halimbahae`  |  31,324 |           3 |
| `tusi`        |  17,421 |           2 |
| `sunnah`      |   1,896 |           1 |
| `fawaz`       |     122 |           3 |

The six non-`sanadset` sources (~117,934 hadiths, ~69,067 `APPEARS_IN`-linked)
are the real curated corpus and look structurally correct. `sanadset`
(650,986 hadiths, **0** collections) is the entire orphan population and is also
the source of the polluted `Narrator` table (132,999 nodes; isnad phrase
fragments stored as narrator names; `STUDIED_UNDER` near-absent at 186 edges).

## Root cause (with code references)

The defect is a **missing parser output**, compounded by a **non-creating edge
match** and **masked by a test fixture**. Three coupled facts:

1. **`parse_sanadset` never emits a collections file.**
   `src/parse/sanadset.py` (`parse_sanadset`) writes exactly three staging
   files — `hadiths_sanadset.parquet`, `narrator_mentions_sanadset.parquet`,
   and `narrators_bio_kaggle.parquet`. It does **not** write a
   `collections_sanadset.parquet`. Every other loaded parser does
   (`fawaz`, `halimbahae`, `lk_corpus`, `sunnah_api`, `sunnah_scraped`,
   `thaqalayn`, `tusi`, `bihar`, `open_hadith` all emit `collections_*`).
   The `Hadith` nodes are still tagged with a `collection_name` derived from the
   CSV filename stem (`collection_name = csv_file.stem.lower()...`), but no
   matching `Collection` is ever produced.

2. **`Collection` nodes load only from `collections_*.parquet`, and `APPEARS_IN`
   uses a non-creating `MATCH`.**
   `_load_collections` (`src/graph/load_nodes.py`) ingests only
   `collections_*.parquet`, so zero `sanadset` `Collection` nodes exist. The
   edge loader `_APPEARS_IN_QUERY` (`src/graph/load_edges.py`) is
   `MATCH (h:Hadith)... MATCH (c:Collection)... MERGE (h)-[:APPEARS_IN]->(c)` —
   a **non-creating** match on the collection endpoint (deliberate: endpoints
   are matched, not merged, to avoid dangling references). `_APPEARS_IN_CHECK`
   reports `collection_exists = false` for every `sanadset` row, so all 650,986
   `APPEARS_IN` edges are skipped as missing endpoints. Result: 650,986 hadiths,
   0 collections.

3. **The canonical-composition gate admits all of `sanadset`.**
   `HADITH_COMPOSITION` (`src/parse/composition.py`) lists per-source dedup
   rules for `halimbahae`, `fawaz`, `mis`, and `open_hadith`. `sanadset` is
   **not listed**, and the default for an unlisted source is "load ALL
   collections" (`is_canonical_hadith` returns `True`). So all 650,986
   `sanadset` hadiths are admitted as `Hadith` nodes with **no** dedup against
   the canonical six Sunni books that are already loaded (with richer metadata)
   via `lk` / `halimbahae` / `fawaz`.

Two corroborating signals in the codebase:

- **The validate baseline codifies the gap.** `DEFAULT_BASELINES`
  (`src/parse/validate.py`) has a `collections_*` baseline for every loaded
  source **except** `sanadset` — and it expects `narrator_mentions_sanadset`
  `row_count = 2,789,517`. That 2.79M-mention firehose, parsed at face value
  from coarse `<NAR>` tags by `_extract_narrator_mentions`, is what pollutes the
  resolved narrator table.
- **A test fixture masks the production gap.**
  `tests/integration/test_sanadset_lightup.py` proves `APPEARS_IN` lands for
  `sanadset` — but only because it imports `write_collections` from the test
  `conftest` and **fabricates** a `sanadset` `Collection` out of band. Production
  has no such step, so the green test never exercised the real (absent)
  collection-emission path. (Cf. the org memory "test-mock injection masks
  production failure".)

Narrator pollution and the near-absent `STUDIED_UNDER` (186) are the same root:
`sanadset` mentions are flat per-`<NAR>` extractions fed straight into narrator
resolution (`_PHASE1_MENTION_SOURCES` in `src/resolve/ner.py`), while
teacher-student edges are derived from transmission/network edges that this flat
corpus does not provide.

## Decision options

### Path A — purge `sanadset` from the live graph (owner-run, irreversible)

**Mechanics.** App-side Admin -> Data Management -> "Danger Zone — Per-source
Purge" deletes the whole `sanadset` source corpus from the live graph (its
`Hadith` nodes, their edges, and `Narrator` nodes reachable only through
`sanadset`). To stop the next `run_all` re-introducing the orphans, pair the
purge with a one-line code change: mark the `sanadset` adapter `active=False` in
`src/adapters.py`, or register `sanadset: frozenset()` in
`src/parse/composition.py` so no `sanadset` `Hadith` nodes load.

**Effort.** Low — one admin action + a one-line registry/composition change and
a fixture-honesty fix to `test_sanadset_lightup.py`.

**Blast radius.** Removes ~650,986 `Hadith` nodes (85% of all hadiths), the bulk
of `NARRATED` edges sourced from `sanadset`, and the `sanadset`-exclusive
(polluted) `Narrator` nodes. The curated six sources (~117,934 hadiths,
~69,067 `APPEARS_IN`) are untouched. Collateral to quantify **before** purging
(use `queries/validation/sanadset_orphan_inventory.cypher`): `TRANSMITTED_TO`
edges and resolved narrators that derive from `sanadset` mentions would also go.

**Reversibility.** Irreversible on **live state**, but the corpus is
deterministically re-acquirable from Mendeley Data (`5xth87zwb5` v4, pinned
SHA-256 in `src/acquire/sanadset.py`). The loss is the graph state, not the
source — a future segmentation pipeline (Path B) could re-load it.

**What is lost.** Any unique `sanadset` content not covered by the curated six
books. `sanadset` spans more collections (via `books.csv`) than the curated
spine, so that breadth is lost until Path B is built.

### Path B — parse, segment, and link `sanadset` (high effort, high risk)

**Mechanics.** (1) Add a `_parse_books` branch to `parse_sanadset` that reads
the **already-downloaded** `books.csv` (acquired alongside `sanadset.csv` —
`_MENDELEY_FILES` in `src/acquire/sanadset.py`) and emits
`collections_sanadset.parquet` (`COLLECTION_SCHEMA`), mapping each hadith's
`book_id` -> book -> collection slug so `collection_node_id` matches.
(2) Register `sanadset` in `composition.py` to dedup the canonical six books
already loaded via `lk` / `halimbahae` / `fawaz`, else double-count.
(3) Improve `<NAR>` segmentation / narrator cleanup to fix pollution.
(4) Fix `test_sanadset_lightup.py` to exercise the real collection-emitting
path instead of `write_collections`.

**Does the source even support it?** Partially. Collection structure **exists**
(`books.csv` + per-hadith `book_id`), so collection linkage is mechanically
feasible. But clean narrator segmentation and cross-edition dedup are **not**
supported out of the box.

**Effort.** High. Requires the cross-edition canonical-identity dedup that
`composition.py`'s own docstring calls a "tracked fast-follow" — never built —
to avoid duplicating the canonical spine; plus narrator re-segmentation R&D.

**Risk.** High. Without cross-edition dedup, linking 650k raw hadiths
double-counts the six books already loaded with richer metadata. The narrator
pollution is intrinsic to the dataset's coarse tagging and only partially
recoverable.

## Recommendation

**Path A — purge `sanadset` from the live graph, paired with deactivating the
`sanadset` adapter in code so it does not re-load.** Rationale:

- The curated six Sunni books are **already** loaded with richer, deduplicated
  metadata via `lk` / `halimbahae` / `fawaz`; `sanadset` is a largely-redundant
  raw corpus.
- The 650k `sanadset` orphans contribute **0** linked collections and pollute
  the narrator table with a 2.79M coarse-mention firehose.
- Path B's prerequisites (cross-edition dedup + narrator re-segmentation) are
  large, not-yet-built efforts to rehabilitate that redundant corpus.
- Path A is **reversible-in-practice**: `sanadset` is deterministically
  re-acquirable from Mendeley if a future pipeline justifies a clean re-load.

This recommendation is **not** self-executing. The purge is owner-run and
irreversible on live state, so the decision is surfaced as an explicit A/B for
the owner to pick. **Do not execute remediation on the basis of this ADR alone.**

## Consequences

**Positive (Path A).** Removes 85% orphan `Hadith` nodes and the bulk of the
polluted narrator table in one action; the live graph becomes a clean view of
the curated corpus; cheap to do and cheap to reverse-via-re-acquire.

**Negative / trade-offs (Path A).** Loses `sanadset`'s collection breadth beyond
the six canonical books until a Path-B pipeline exists; some `TRANSMITTED_TO`
edges and resolved narrators derived from `sanadset` mentions are removed
(quantify first via the inventory query).

**If Path B is chosen instead.** No data is lost, but it gates a clean switch-over
on cross-edition dedup + narrator re-segmentation — neither built yet — and risks
compounding duplication of the canonical spine if shipped without the dedup.

## Related

- da#202 — keystone: `sanadset` orphan nodes + polluted narrator table (prod).
- noorinalabs/noorinalabs-main#723 — meta: corpus/graph data-quality (Phase 7).
- da#153 — orphan-scale evidence.
- da#191 — canonical corpus composition (`src/parse/composition.py`).
- `queries/validation/sanadset_orphan_inventory.cypher` — read-only blast-radius
  inventory to run before any purge.
