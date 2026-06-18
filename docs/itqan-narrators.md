# Itqan narrator profiles (da#92a)

The **Itqan** corpus (github [`R3GENESI5/Itqan`](https://github.com/R3GENESI5/Itqan))
is the largest open-source narrator (rijal) database: **115,735 narrator
profiles** drawn from 22 classical texts, **72.6 %** of them carrying a
jarh-wa-ta'dil grade. This slice (da#92a, PR 1 of 3 in the Itqan pre-split)
loads the **narrator profiles → `Narrator` nodes**. It is the keystone source
for isnad-graph narrator search (isnad-graph#963 / #965).

The two sibling Itqan PRs cover the rest of the corpus and are **out of scope
here**:

- **#93** — `teachers` / `students` id lists → isnad chains / `TRANSMITTED_TO`.
- **#94** — `namings` / `by_name.json` → narrator name variants / aliases.

## Source shape

`app/data/rijal/manifest.json` enumerates one JSON file per grade bucket:

| bucket file | grade | profiles |
|---|---|---|
| `profiles_reliable.json` | reliable | 26,467 |
| `profiles_mostly_reliable.json` | mostly_reliable | 21,800 |
| `profiles_weak.json` | weak | 21,413 |
| `profiles_companion.json` | companion | 10,880 |
| `profiles_unknown.json` | unknown | 31,695 |
| `profiles_fabricator.json` | fabricator | 2,094 |
| `profiles_abandoned.json` | abandoned | 1,386 |

Each file is a JSON object keyed by narrator id (globally unique across buckets).
The grade bucket = the per-profile `grade_en` field; the Arabic verdict is in
`grade_ar`.

### Field mapping (→ `NARRATOR_BIO_SCHEMA`)

| Itqan field | bio field | notes |
|---|---|---|
| `id` | `external_id`, `bio_id`=`itqan:{id}` | globally unique |
| `full_name` | `name_ar` | Arabic-only source — `name_en` is null |
| `grade_en` | `trustworthiness` | reliable→thiqa, mostly_reliable→saduq, weak→daif, abandoned→matruk, fabricator→kadhdhab, companion→thiqa, unknown→unknown |
| `grade_ar` | `bio_text` | the jarh-wa-ta'dil verdict text |
| `death` | `death_year_ah` | free text ("بين 161 هـ إلى 170 هـ"); first Hijri year taken as an **approximate** anchor |
| `tabaqat` / companion bucket | `generation` | companion→sahabi; a conservative Arabic-ordinal map otherwise, else null |
| `city` | `birth_location` | primary city of activity |
| `kunya`/`nasab`/`laqab` | `kunya`/`nisba`/`laqab` | `-` placeholder → null |

`gender` and `birth_year_ah` are not recorded by Itqan → null.

## Why a bio-direct promoter

The mention-driven disambiguator (`src/resolve/disambiguate.py`) only mints a
canonical `Narrator` when an **isnad mention** resolves to a bio candidate — so a
profile-only source (bios, no chains in this slice) would yield **zero** Narrator
nodes. `src/resolve/bio_promote.py` bridges that: each bio is promoted to a
canonical narrator keyed by the **same** `nar:<uuid5(normalized-name)>` identity
the disambiguator uses (`src.parse.identity.make_canonical_id`), so the two paths
converge — a later mention-driven run dedups onto the same node — and the graph
loader ingests the output unchanged. It is merge-safe: an existing
`narrators_canonical.parquet` is unioned, not overwritten.

> The graph loader (`load_nodes._load_narrators`) reads
> `narrators_canonical.parquet` from the **staging** dir, so the promoter must be
> pointed at staging for the load to pick it up.

## Full-load runbook (all 115,735 profiles)

```bash
# 1. Acquire — downloads the 7 profile buckets (~150 MB) into data/raw/itqan/
uv run python -m src.cli acquire        # or: only this source via src.acquire.itqan.run(raw_dir)

# 2. Parse — narrators_bio_itqan.parquet (NARRATOR_BIO_SCHEMA)
uv run python -m src.cli parse

# 3. Promote bios → narrators_canonical.parquet IN the staging dir
uv run python - <<'PY'
from pathlib import Path
from src.config import get_settings
from src.resolve.bio_promote import promote_bios_to_canonical
s = get_settings()
promote_bios_to_canonical(Path(s.data_staging_dir), Path(s.data_staging_dir), sources={"itqan"})
PY

# 4. Load Narrator nodes into Neo4j
uv run python -m src.cli load --nodes-only
```

### Representative live proof (committed)

`tests/integration/test_itqan_load.py` loads a committed 24-profile real sample
(abandoned + fabricator + companion) into a live `neo4j:5` container and asserts
`nar:` ids, the `external_id` provenance tag, and the three jarh tiers. A larger
local proof run (3 real buckets, 14,360 bios) promoted to **12,820 canonical
narrators** and loaded **12,820 `Narrator` nodes** into `neo4j:5` with 0 errors
(trust split: thiqa 9,922 · kadhdhab 1,639 · matruk 1,259).

## Reachability & licensing

- **Reachability:** plain HTTPS from `raw.githubusercontent.com` — no auth, no
  API key. The 7 buckets total ~150 MB.
- **Licensing:** the upstream repo ships **no LICENSE file** → all-rights-reserved
  by default. We do not redistribute their files — we ingest **facts** (names,
  grades, dates) re-expressed in our own schema, with full provenance. Cleared by
  the owner for this non-profit use on that basis (upstream author: Ali Bin Shahid).

### Provenance & removability

Every `Narrator` node carries `source_ids` (a `<corpus>:<bare-id>` list, e.g.
`["itqan:320"]`), so the Itqan contribution is auditable and **cleanly removable
on the graph** if the upstream author ever objects:

```cypher
// remove every narrator that came (solely) from Itqan
MATCH (n:Narrator) WHERE any(s IN n.source_ids WHERE s STARTS WITH 'itqan:')
DETACH DELETE n
```

> `source_ids` (a list) is the correct removal handle rather than a scalar
> `source_corpus`: after cross-source dedup a canonical narrator can carry ids
> from several corpora, so to *strip* Itqan from a multi-source narrator you would
> remove its `itqan:` entries rather than delete the whole node.

## Staging load (da#176) — executed 2026-06-16

The full 115,735-profile load onto **staging** Neo4j, the narrator-side coverage
keystone for the P5W5 production cutover (`deploy#470`).

### What landed (read back from staging)

`bio_promote` collapsed the 115,735 bios to **85,840** canonical narrators keyed
by `nar:<uuid5(name_ar_normalized)>` (`make_canonical_id`) — many profiles share a
normalized name. Verified counts on the live staging graph:

| Metric | Value | Note |
|--------|-------|------|
| Total `Narrator` nodes | 132,999 | was 47,199 → **+85,800** new Itqan |
| `source_corpus = itqan` | 85,840 | 85,800 Itqan-only + 40 unioned (see below) |
| `source_corpus = lk` | 47,089 | **unchanged — 0 `lk` nodes touched** |
| `source_corpus = muhaddithat` | 70 | was 110; 40 re-derived primary corpus |
| Itqan provenance (`source_ids` has `itqan:`) | 85,840 | the removal handle above |
| Non-`nar:` ids | 0 | every node keeps the canonical identity |

Itqan trustworthiness split: thiqa 29,605 · unknown 21,869 · saduq 17,759 ·
daif 13,709 · kadhdhab 1,639 · matruk 1,259.

### No-clobber merge of the 40 collisions

40 of the 85,840 Itqan canonical ids collided with already-loaded **muhaddithat**
narrators (both sources are bio-promoted by normalized name, so they converge on
the same `nar:` id — exactly the intended cross-source dedup). A naive load would
have let `_NARRATOR_MERGE`'s unconditional 15-property `SET` clobber those nodes'
`mention_count` and prior `source_ids` — the two-producer hazard
(`disambiguate` OVERWRITES, `bio_promote` MERGES). It was avoided by feeding the 40
existing canonical records back through `promote_bios_to_canonical`, whose
parquet-level union preserves `mention_count` and unions `source_ids` /
`source_corpora` before the graph `MERGE`. Read back after the load:

```text
nar:200b9c92-1ea1-5336-831f-b8bc3a773a19  أبو موسى الأشعري
  source_ids      = ["muhaddithat:41", "itqan:74959"]   # prior id preserved + Itqan added
  source_corpora  = ["itqan", "muhaddithat"]
  source_corpus   = "itqan"                              # primary_corpus() is alphabetical-first
  mention_count   = 0                                     # preserved, not reset
  sect_affiliation= "neutral"                             # muhaddithat cross-tradition, preserved
```

The scalar `source_corpus` reads `itqan` for the 40 only because `primary_corpus()`
takes the alphabetically-first corpus over `{itqan, muhaddithat}`; the full
multi-source truth lives in `source_corpora` + `source_ids`. This is the identical
result a single `run_all` (`disambiguate` → `bio_promote`) over both sources would
produce.

### Reproducing the produce step

The acquire → parse → promote half is the committed, deterministic
[full-load runbook](#full-load-runbook-all-115735-profiles) above; it writes
`narrators_canonical.parquet` (85,840 rows, `source_corpus=itqan`) with no Neo4j
access.

### The load mechanism

Staging Neo4j (`bolt://neo4j:7687`) is reachable only **inside the cluster** and
the staging container has `apoc.import.file.enabled=false`, so the load ran on the
staging box via `docker exec … cypher-shell` against the
`noorinalabs-neo4j-1` container — the same out-of-band channel prior staging loads
used (da#73 / da#141). The executed Cypher replicates `_NARRATOR_MERGE` verbatim
(idempotent `MERGE` on `id`; re-running is a no-op), and the 40-record union above
runs through the committed `promote_bios_to_canonical`. The **go-forward canonical
loader** is the containerised `noorinalabs-graph-load` image (da#174,
`Dockerfile.load` + `scripts/load_staging.sh`), which runs `graph.load_all` inside
the cluster where bolt resolves; once merged, prefer it over the manual channel.
