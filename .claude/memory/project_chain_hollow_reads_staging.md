---
name: Chain nodes hollow — read staging not curated mentions
description: _load_chains read raw staging mentions (no canonical_narrator_id) → all chains hollow; fix = curated resolved master. #723 "chains empty" root cause.
metadata:
  type: project
---

The #723 "chains empty" prod symptom root cause: `src/graph/load_nodes.py::_load_chains`
built Chain nodes by reading `narrator_mentions_*.parquet` from **staging/** and pulling
`canonical_narrator_id` from each row — but the **staging** mentions have NO
`canonical_narrator_id` column (it is produced by the resolve stage and written ONLY to
`curated/narrator_mentions_resolved*.parquet`). So `row.get("canonical_narrator_id")`
was always `None` → every chain got `narrator_ids=[]`, `chain_length=0`. All 519k chains
hollow. The function's own docstring said "from resolved data" — code read the wrong file.

**Fix (commit 4e20597):** point `_load_chains` at the curated resolved master (mirrors
`_load_narrators`, which already takes `curated_dir`); accumulate only `(position,
canonical_id)` per hadith to stay memory-lean over the 3.2M-row file. `scripts/
load_staging.sh` must ship curated resolve outputs for **nodes-only** loads too — Chain is
a node type that legitimately needs resolve output (the old transport gated curated behind
the full-load branch only).

Validated on stage: chains **0% → 99.999% populated**, avg length 5.5, 97.9% at a plausible
isnad length (1–12); only 0.07% implausibly long (>30), the known thaqalayn/sanadset hadith-id
collisions. Schema gotcha that wasted a query: Narrator names are `name_ar`/`name_en` (NOT
`canonical_name`); Chain carries narrators as a `narrator_ids` array PROPERTY + `hadith_id`,
not as edges — so an edge-connectivity query is the wrong lens for chain population. Verify
via `c.chain_length` / `size(c.narrator_ids)`. Related: [[project_staging_graph_load_transport]],
[[feedback_count_ge_zero_masks_empty_graph]].
