---
name: project_name_normalized_provenance
description: name_ar_normalized is WRITTEN by clean_narrator_name(normalize_arabic(x)) but RE-DERIVED downstream by normalize_arabic(x) alone — 47,586/129,234 rows drift (1.7% mention-weighted). normalize_arabic IS idempotent; don't blame it. da#371.
metadata:
  type: project
---

`name_ar_normalized` / `name_normalized` has **two producers with different definitions**, so the stored value and any re-derived value disagree.

| | function |
|---|---|
| **written as** | `clean_narrator_name(normalize_arabic(x))` — `src/parse/name_quality.py` (da#247 family) |
| **re-derived at** | `src/resolve/disambiguate.py:629` — `normalize_arabic(x)` **alone** |

`make_canonical_id` keys on whichever string it is handed ⇒ **two canonical ids for one name**.

## Do NOT blame `normalize_arabic`

It is **provably idempotent**: `f(f(x)) == f(x)` over all 129,234 canonical names, **0 counterexamples**. It also strips bidi/zero-width marks as step 1 (`src/utils/arabic.py:51`, added by da#271 `7e28c1c` 2026-07-03). I filed the "non-idempotent" diagnosis first and it was **wrong**; the team lead caught it by testing the function against the corpus.

The disambiguating evidence is a **non-Arabic** row, which rules out any Arabic-normalization story:

```
name_ar                                        = 'Khabbab b. al-Aratt'
stored name_ar_normalized                      = 'Khabbab b al-Aratt'    # '.' dropped
normalize_arabic(name_ar)                      = 'Khabbab b. al-Aratt'   # '.' PRESERVED
clean_narrator_name(normalize_arabic(name_ar)) = 'Khabbab b al-Aratt'    # exact match
```

## Scale — quote both numbers or you will mislead

| quantity | value |
|---|---|
| canonical rows where stored `!= normalize_arabic(name_ar)` | 47,586 / 129,234 (**36.8%**) |
| ...stored is a strict **prefix** of re-derived (truncation) | 18,255 |
| ...`mention_count == 1` | 20,134 |
| **mention-weighted drift** | 53,933 / 3,249,721 (**1.7%**) |
| mention surfaces carrying bidi marks | 215 / 241,459 |
| `narrators_bio_kaggle.parquet` un-normalized rows | 12 |

**36.8% is a row count, not a graph-scale defect.** Weighted it is 1.7% and mostly singletons — the OCR/matn tail ([[project_matn_sentence_pollution_ui]]) seen from the other side.

Second-order: `_load_candidates` (`disambiguate.py:387`) reads the bio `name_ar_normalized` column **without re-normalizing**, so those 12 kaggle rows sit in the Stage-1 exact-match blocking index under keys no clean mention can ever match — silent recall loss.

Tracked by **da#371**. da#356 / PR #363 re-keys identity on `normalize_arabic(mention_text)`, making `disambiguate` internally consistent with its own matcher, but does **not** repair the column. #316's stated mechanism (RLM shielding the edge-punct strip) is stale for the same reason.

Fix direction: one authority for the column, every producer *and* consumer routed through it — the same rule as [[feedback_dual_detector_cross_sect_authority]]. See also [[project_narrators_two_producers]] (disambiguate OVERWRITES, bio_promote MERGEs) and [[project_relational_pollution_scrub_equiv]].
