---
name: project_canonical_identity_invariant
description: da#356 — canonical id/display name is a function of the MENTION, never the matched bio. crossref (Stage 5), not fuzzy, caused 92% of the 14,316 chimeric nodes. Corroboration criterion "mention is bio-registered" is vacuous by stage ordering. PR #363.
metadata:
  type: project
---

**The invariant** (`src/resolve/disambiguate.py`, PR #363):

> `norm_name ∈ { mention_norm, refine_mononym_name(mention_norm, chain_evidence).norm_name }` — and **never** `candidate.name_ar_normalized`.

A bio match may attach `external_id`, `birth_year_ah`, `death_year_ah`, `generation`, `gender`, `trustworthiness`, `source_ids`. **It may not rename and may not re-key.** Enforced at the sink: `_upsert_canonical` never back-fills `name_ar`/`name_en` from the candidate.

Pre-fix, a mention `عائشة` fuzzy-matching the OCR-corrupt itqan bio `عائذة` produced a node keyed `uuid5(عاءذه)` displayed `عاءذه`, with the correct spelling demoted to `aliases`. **14,316 chimeric nodes / 593,456 mentions.**

## Three things that are counter-intuitive and cost time

1. **`crossref` (Stage 5) is the mechanism, not `fuzzy` (Stage 2).** crossref = 19,900 of 21,644 chimeric nodes (**92%**), 474,126 of 612,913 mentions. `fuzzy` needs `ratio≥80 ∧ lev≤2` then runs `_temporal_filter` + `_geographic_filter`; crossref needed only `ratio≥60`, had **no Levenshtein bound**, and ran through **neither filter**. With a 2-char blocking prefix every `ابو …` kunya falls in one block — one node accreted **607 distinct names**, merging Imam al-Bāqir with `ابو نصر احمد بن سهل الفقيه`.

2. **"Corroborate by requiring the mention's own form to be bio-registered" is VACUOUS.** Survival 0/742,607 fuzzy, 0/937,539 crossref — **0.000%**, provable a priori: Stage 1 `_exact_match_indexed` early-returns on any exact bio hit, so nothing reaching Stage 2/5 can be bio-registered. Adopting it silently deletes both stages.

3. **Requiring chain-neighbour temporal agreement alone rejects on MISSING DATA.** Only **26.2%** of the 140,174 bio candidates carry a `death_year_ah`. Shipped gate = *no temporal contradiction* **AND** (*evidence exists and agrees* **OR** `score≥0.90 ∧ lev≤1`).

The gate matters even though identity is now safe: an attached `death_year_ah` enters `death_year_index` → feeds `_temporal_filter` for neighbours **and** `refine_mononym_name`'s da#248 evidence → **bad metadata propagates back into identity.**

## De-keying does NOT fragment the graph

Node count 34,915 → 162,349 at disambiguate output (4.65×). Of the 128,058 new nodes, classified against each group's modal form: **10.6% spelling variants** (`fuzzy_cluster` re-merges; `_choose_representative` ranks by `mention_count`) vs **89.4% genuinely different people**. The shatter is an over-merge being undone. 63.9% of post-fix forms are `mention_count==1` singletons — the OCR/matn tail, zero graph-metric weight.

## Gotchas for anyone touching this

- **The da#248 carve-out is load-bearing.** `refine_mononym_name` *intentionally* re-keys off the mention surface using chain evidence. A naive "a node's name must appear among its own mentions" assertion false-positives on every mononym split (3 nodes) — and an assertion that fires on correct behaviour gets deleted by the next person.
- **`MONONYM_REGISTRY` abstains unless evidence fits EXACTLY ONE person.** Writing a da#248 test with neighbour year 131 proves nothing: it is plausible for al-Thawrī (d.161, gap 30) *and* ibn ʿUyayna (d.198, gap 67). Derive a discriminating year; don't assume one.
- `_CONFIDENCE_THRESHOLD` (0.70) was **never** the defect — the identity *source* was. Post-fix it governs only whether metadata attaches.
- `أنس بن مالك`/`أنس` (da#347) and bare `عبد الله` (da#346) were already correctly self-keyed; this fix does not address them (they are `narrator_split`/`generic_name` territory).
- Identity keys on `normalize_arabic(mention_text)`, not the stored column — see [[project_name_normalized_provenance]] (da#371).

Related: [[project_narrators_two_producers]], [[project_matn_sentence_pollution_ui]], [[project_relational_pollution_scrub_equiv]].
