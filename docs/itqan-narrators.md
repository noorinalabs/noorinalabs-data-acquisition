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
  by default. Acquisition here is for research/analysis. Redistribution or
  production publication of the raw corpus needs an explicit usage grant from the
  upstream author (Ali Bin Shahid) — **flagged for owner confirmation; not
  cleared for redistribution.**
