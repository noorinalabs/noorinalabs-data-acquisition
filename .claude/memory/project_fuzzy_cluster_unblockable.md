---
name: project_fuzzy_cluster_unblockable
description: da#376 — fuzzy_cluster blocks on combinations(significant_tokens,2), so a name with <2 significant tokens is NEVER scored at any threshold. 7,694 forms / 36.2% of merge_log mentions. Structural, not statistical. Cure = fold inflection inside make_canonical_id, not tuning.
metadata:
  type: project
---

`fuzzy_cluster.cluster_records` builds blocks from `itertools.combinations(significant_tokens, 2)` and `_can_merge` requires `_MIN_SHARED_TOKENS = 2`. `ابو`, `ابي`, `ابن`, `بن`, `ام`, `ال`, `بنت`, `عن` are all in `_CONNECTOR_TOKENS`.

**So a name with fewer than two *significant* tokens produces zero blocking keys, joins no block, and is never offered to `_can_merge` — at any threshold, by any tuning.** Structural, not statistical. Measured on the pre-fix production `merge_log`: **7,694 forms holding 1,065,833 mentions (36.2%)** had never once been scored.

This was latent and harmless while `disambiguate` keyed identity on the matched **bio's** spelling — the bio key incidentally absorbed Arabic inflection. da#356 correctly made identity a pure function of the mention and handed the job to `normalize_arabic`, which is neither case- nor spacing-invariant. **Over-merge became under-merge**, on the most-cited narrator in the corpus:

```
ابي هريره  46,563     ابا هريره  5,691     ابو هريره  2,179     = 54,433
pairwise _can_merge = False for all three pairs
```
(note `ابا` is *not* in `_CONNECTOR_TOKENS` while `ابو`/`ابي` are, so the three forms don't even fail symmetrically). Same for `انس`/`انسا`, `علي`/`عليا`, `مالك`/`مالكا`, `جابر`/`جابرا`.

## The cure is a fold before minting, not a threshold

`canonical_surface` (`src/utils/arabic.py`) is applied **inside `make_canonical_id`**, not at the call site — there are **eight** id-minting sites (`disambiguate`, `bio_promote`, `narrator_split`, `date_reconcile`, `muhaddithat_links`, `fuzzy_cluster`, `load_edges`, `make_discriminated_canonical_id`) and a fold applied in only one reintroduces duplicates by another door.

**PRECISION-FIRST — every rule is a closed set or a lexicon, never a productive orthographic pattern.** Measured on the corpus, a naive "strip a trailing alif from any token ≥4 chars":
- folds `زكريا` (Zakariyyā, **9,207 mentions**) → `زكري`
- folds `عمرا` (accusative of `عمرو`, ʿAmr) → `عمر` (ʿUmar) — **two different men**

No orthographic rule separates `عليا`/`علي` from `زكريا`/`زكري`; only a lexicon does. Likewise the `عبد` split must trigger on the **article** (`عبدال`), never a bare `عبد` prefix: `عبدان` (3,974), `عبدوس`, `عبدويه` are names in their own right.

## Residual, and how to talk about it

The fold reduces the never-scored population 7,694 → 7,169 forms; it does **not** close the structural gap. `_MIN_SHARED_TOKENS = 2` is a deliberate precision guard ("a bare single-token subset never clusters"). PR #363 makes the gap *countable* — `ClusterMetrics.unblockable_records` + a `cluster_records_unblockable` warning — because previously "never offered to `_can_merge`" and "`_can_merge` said no" were the same observable outcome: no merge, no log, no counter. Same success-shaped silence as da#309.

**Never assert a merge outcome from a lexical proxy.** A `ratio>=90 & lev<=2` heuristic labels `الحسين` vs `الحسن` (Ḥusayn vs Ḥasan — brothers) and `عمرو بن ابي سلمه` vs `عمر بن ابي سلمه` as "variants". Run `_can_merge`. Derived fates of the 124,834 new nodes de-keying creates: 16.6% will re-merge, 77.6% scored-and-refused, 5.8% never scored.

Related: [[project_canonical_identity_invariant]] (da#356), [[project_fuzzy_cluster_throughput]] (da#270 caps), [[project_name_normalized_provenance]] (da#371).
