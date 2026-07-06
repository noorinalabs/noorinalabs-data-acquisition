---
name: project_matn_sentence_pollution_ui
description: Narrator set has ~26% matn-sentence pollution in the zero-degree tail; da#311 scrub + degree-based validation missed it (da#317). Validate on the UI surface, not just top-by-degree.
metadata:
  type: project
---

Post-#723 staging reload (clean scrubbed data, 150,187 canonical narrators), the **narrator entity set still carries a large matn-sentence / Qur'anic-verse pollution class** — full sentences mis-extracted as narrator names, living in the **zero-degree orphan tail**. Filed **da#317** (data-quality). Distinct from [[project_relational_pollution_scrub_equiv]] (single-token pronouns), da#314 (trailing-isnad leakage), da#315 (ما/وبه residual) — this is whole *sentences*.

**Scale (staging Neo4j, post-reload):** zero-degree orphans 45,617 (30.4%); of those, matn-sentence-shaped (≥5 words) 38,763 (**25.8%**); conjunction-initial (و/ف, ≥4 words) 10,958; orphans w/ قال/كان/الله 7,009. Example row-1 of id-ordered `/narrators`: `الله ومن يعظم شعاءر الله فانها من تقوى القلوب` (Qur'an 22:32).

**Not a reload regression** — pre-existing; da#311 scrub (165,939→150,187) removed single-token junk + isnad residue but never addressed matn *sentences*. Old graph had more of it.

**Why da#311 validation falsely passed — two blind spots (the lesson):**
1. **Degree-based checks miss the orphan tail.** Top-narrators-by-degree were clean because matn orphans are zero-degree and never rank. The pollution is invisible unless you inspect the *id-ordered* list — i.e. the **UI surface the user actually sees**. Validate there, not just via `ORDER BY degree` Neo4j probes.
2. **Hamza-form false-0.** Probed `شعائر` (hamza ئ) but stored `name_ar` is `normalize_arabic`-folded to `شعاءر` (bare ء) → `CONTAINS` returned 0. `normalize_arabic` folds ئ→ء (and أ→ا) but NOT ى→ي. **All Arabic validation probes must use normalized hamza forms** (see [[project_relational_pollution_scrub_equiv]]).

The load-bearing UI field is `name_ar` (served by `/api/v1/narrators`, source = Neo4j, NOT Postgres — pg `isnad_graph` schema holds only `hadiths`/`hadith_embeddings`). Narrator list total (150,187) is the full unfiltered set — no zero-degree filter.

**Remediation levers:** (1) quick UI mitigation = filter zero-degree orphans from `/narrators` (drops ~30%, nearly all junk) — isnad-graph; (2) root fix = extend name_quality gate to a matn-*sentence* rule class + recall guard for legit long-nasab, re-scrub, re-load, **re-validate on the UI surface** (da#317).

**Sibling UI-review finds (same session):** da#318 — hadith matn served with raw `<NAR>/<SANAD>/<MATN>` markup (parser leak; `hdt:sanadset:0:0:107270`); ig#1166 — Graph Explorer ignores `?narrator=` deep-link (renders id-first node). App otherwise functional (search via `q=` works, ego-graph renders ~200 nodes, no console errors). **UPDATE 2026-07-06: prod WAS promoted** (#723 closed, purge+reload of the same 150,187 scrubbed artifact, record-level verified at exact stg parity) — the zero-degree matn-sentence tail (orphans 44,073 on both stg+prod) rode along and is now the **da#317 carry-forward**, NOT a blocker: #723's acceptance criterion is matn-opener pollution **weighted by mention_count**, and this tail is mc≤1/zero-degree so it weights ~0.000%. So the closure is honest AND this tail is real work still owed (da#317). See [[project_disambiguate_checkpoint_resume]], [[project_fuzzy_cluster_throughput]].
