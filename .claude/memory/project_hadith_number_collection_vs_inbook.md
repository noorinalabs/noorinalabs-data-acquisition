---
name: project_hadith_number_collection_vs_inbook
description: "sunnah.com hadiths carry TWO numbers — a collection-wide ref (e.g. 680, for source_id uniqueness) and an in-book ordinal (Book 1 Hadith 1). The APPEARS_IN edge prop hadith_number_in_book means the in-book ordinal, NOT the collection ref."
metadata: 
  node_type: memory
  type: project
  originSessionId: 080813cd-f3b8-434d-974c-badf58620c96
---

sunnah.com pages (verified on riyadussalihin/1) show **two distinct numbers**
per hadith:
- **collection-wide reference** — "Riyad as-Salihin 680". Globally unique within
  the collection. Belongs in `source_id` (`sunnah:<coll>:<book>:<chapter>:<ref>`).
  da#72 (#75, Alejandra) extracts THIS into the staging `hadith_number` field.
- **in-book reference** — "Book 1, Hadith 1". The ordinal within the book. This
  is what the `AppearsIn` model's `hadith_number_in_book` property means — its
  docstring is literally "Hadith number within the book".

**The trap (Oyunbileg-flagged, settled in da#77):** the APPEARS_IN edge loader
maps `row.hadith_number` → `hadith_number_in_book`. After da#72 that staging
field holds the **collection-wide ref (680)**, so the edge prop would carry the
wrong semantic value (680 where the name promises the in-book ordinal 1).

**Decision (Kwesi, da#77, 2026-06-11):** `hadith_number_in_book` must carry the
**in-book ordinal**; the collection-wide ref stays in `source_id` only. The prop
name is already correct — it needs the right value, not a rename. Populating it
requires the SCRAPER to extract the in-book ordinal into its own staging column
(da#72/Alejandra's parser lane, not the edge loader). Proposed split: da#77 ships
the null-safe MERGE+SET (prop null until the field exists); da#72 follow-up
extracts the in-book number into a dedicated column. See
[[feedback_appears_in_merge_null]].
