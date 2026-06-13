# Narrator-graph produce + load (da#141)

The **data half** of `main#601` criterion #1: produce and load the REAL narrator
graph — `Narrator` + `NARRATED` + `STUDIED_UNDER` + `APPEARS_IN` +
`TRANSMITTED_TO`, **both sects** — onto staging Neo4j.

It is the narrator-graph successor to the da#73 [first-light slice](first-light-slice.md),
which proved only the hadith + `APPEARS_IN` path on one collection. The gap
#601 found was: staging held 47 Hadith + 47 `APPEARS_IN` and a **zero narrator
graph** (0 `Narrator`, 0 `NARRATED`, 0 `STUDIED_UNDER`) — the W4-retro
"data-first core shipped" was local/CI/harness only, never reflected on staging.

The driver is `scripts/narrator_graph/run.py`:

```
acquire -> parse -> resolve.run_all -> graph.load_all -> verify
```

## The produce step (reproducible)

The default source set is **bounded but REAL** (not toy fixtures), so the whole
produce → load → verify path runs locally end-to-end (the fuzzy-cluster step
over the full ~47k `lk` narrators is the long pole — several minutes, longer
under CPU contention; it is the da#118 recall pass and does not affect load
correctness):

| Source        | Sect       | What it contributes |
|---------------|------------|---------------------|
| `muhaddithat` | both (neutral) | Cross-tradition female narrators. No isnad text, so its narrators reach canonical via `bio_promote` (**bio-only promoted**, tagged `neutral`); its studentship pairs → `STUDIED_UNDER`. |
| `lk` (LK-Hadith-Corpus) | sunni | 6 Sunni books with isnad chains → hadiths + narrator mentions → `NER`/`disambiguate` → canonical narrators → `NARRATED` + `TRANSMITTED_TO` + `APPEARS_IN`. |

Together they exercise **every** narrator-graph edge type AND both sects
(`lk` sunni mention-derived + `muhaddithat` neutral bio-only narrators), the
da#120 disambiguate→bio_promote ordering, and the da#133 per-row `relation`
`STUDIED_UNDER` routing.

```bash
# Produce only (acquire + parse + resolve); no Neo4j. Writes data/curated/.
uv run python scripts/narrator_graph/run.py --produce-only
```

Widen the set with `--sources` for the full staging load — e.g. add
`thaqalayn` (Shia Four Books, for Shia hadith coverage + cross-sect
`PARALLEL_OF`) or `itqan` (the 100k-narrator rijal DB).

## Local verification (the `neo4j:5` container)

```bash
# 1. Local Neo4j (WSL2 docker now works — memory: wsl2-no-local-docker RESOLVED).
docker run -d --name da141-neo4j -e NEO4J_AUTH=neo4j/testpassword123 \
    -e NEO4J_PLUGINS='["apoc"]' -p 7688:7687 neo4j:5

# 2. Produce + load + verify against it.
NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword123 \
    uv run python scripts/narrator_graph/run.py
```

The driver loads, then runs Cypher counts and asserts the required invariants
(non-zero `Narrator` / `STUDIED_UNDER` / `NARRATED` / `APPEARS_IN`, a bio-only
promoted narrator present, both sects present). It exits non-zero if any fails
(`--no-assert` to report-only).

### Verified counts (default `muhaddithat` + `lk` set, 2026-06-13)

Loaded into a local `neo4j:5` container; **idempotent** on re-load (`MERGE` on
`id` — a second load creates 0 new nodes/edges, counts unchanged):

| Node / edge | Count | Note |
|-------------|-------|------|
| `Narrator`  | 47,199 | sunni 47,086 (`lk`, mention-derived) + neutral 113 (`muhaddithat`, **bio-only promoted**) |
| `Hadith`    | 34,088 | `lk` |
| `Collection`| 6 | `lk` books |
| `NARRATED`        | 33,977 | first narrator per chain → hadith |
| `TRANSMITTED_TO`  | 52,182 | consecutive narrator pairs |
| `APPEARS_IN`      | 33,981 | incl. **39 with a null in-book ordinal** — proves the ip#84/da#77 null-safe MERGE |
| `STUDIED_UNDER`   | 186 | `muhaddithat` studentship; both-sect neutral narrators; 0 missing endpoints |
| `GRADED_BY`       | 33,478 | `lk` grades |
| `PARALLEL_OF`     | 0 | none in a single-sect set — appears once a Shia hadith source (e.g. `thaqalayn`) is added |

> **Loader fix shipped with this work (da#141):** `load_all_edges` read
> `narrator_mentions_resolved.parquet` from **staging**, but `resolve.run_all`
> writes it to **curated**. The chain edges silently fell back to the raw,
> canonical-id-less staging mentions → **0 `NARRATED` / `TRANSMITTED_TO`** on a
> real orchestrated load (exactly the #601 staging gap). The fix wires
> `curated_dir` into both chain-edge loaders; covered by
> `TestChainEdgesReadResolvedFromCurated` in `tests/test_graph/test_load_edges.py`
> and by the live counts above.

## Staging load — GATED

> **Do NOT load live staging from a dev worktree.** This write pairs with the
> ingest-platform worker bring-up (ip#83) and is the gated `#601` run the
> team-lead coordinates.

Staging Neo4j (`bolt://neo4j:7687`) is reachable only **inside the cluster**
(prior loads — da#73 — used `docker exec cypher-shell` on the box). Run the
driver from a box that resolves it, with the staging credentials in the env:

```bash
export NEO4J_URI=bolt://neo4j:7687          # staging bolt endpoint (cluster-internal)
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<the deployed NEO4J_PASSWORD secret>

# Produce on the box (or copy a produced data/ dir over), then load + verify.
uv run python scripts/narrator_graph/run.py
```

The loader is idempotent (`MERGE` on `id`), so re-running over the existing 47
da#73 hadiths is safe — they `MERGE`, not duplicate. Expect the verify block to
report the counts above (plus the pre-existing 47 riyadussalihin `APPEARS_IN`).

### Acceptance for the gated run

`main#601` criterion #1 is met when, on **staging**, the verify block reports:
non-zero `Narrator`, `STUDIED_UNDER`, and `NARRATED`, both sects present, and a
bio-only promoted narrator — i.e. the same invariants the local run asserts,
now live on the staging Neo4j.
