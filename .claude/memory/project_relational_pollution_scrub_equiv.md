---
name: Relational-pronoun narrator pollution + scrub equivalence
description: da#247 residual — أبيه "his father" was #1 narrator; filter in name_quality; scrubbing curated artifacts is provably equivalent to NER-with-filter (singleton clusters).
metadata:
  type: project
---

da#247 residual found validating the #723 reload: the first name-quality pass killed the
collective-phrase mubham form (leader-token + partitive `من`) but left the **single-token
relational-pronoun** form — `ابيه` "his father" (65,755 mentions, the **#1 "narrator"**),
`جده` "his grandfather", `ابي` "my father", `اخيه`/`جدته`/`خاله`/… They pass as valid
1-token Arabic and carry no `من`, so the collective guard misses them. Each is a spurious
node every chain's *elided ancestor* collapses into, fabricating cross-chain links.

**Fix (commit 4e20597, `src/parse/name_quality.py`):** `_MUBHAM_RELATIONAL` set, dropped only
when the **whole** cleaned name is exactly one such token (precision-safe — `ابي اسحاق` =
Abū Isḥāq survives). Also strip edge punctuation per token, else a trailing Arabic comma/colon
(`ابيه،`, `ابي ،`) shields the bare token (caught 90 extra). Removes 1,326 narrators / 83,038
mentions; top narrator becomes legitimately `ابو هريرة`.

**Scrub equivalence (key technique):** `scripts/scrub_relational_pollution.py` applies the
filter to ALREADY-built curated artifacts (drop rejected narrators_canonical rows + drop
mentions referencing dropped canonical ids) instead of re-running NER→disambiguate→enrichment.
This is **provably equivalent** to re-running NER-with-filter, not a heuristic: relational
names cluster only with themselves (disambiguate keys on normalized name → singleton canonical),
so removing their mentions cannot change any surviving narrator's clustering. NER drops a
filtered span entirely (`dropped += 1; continue`), so a mention-row drop matches exactly. Use
the scrub for fast stage rehearsal; the committed filter is the durable fix for the next real
resolve run (incl. prod).

**Still open under da#247:** English non-name fragments in `name_ar` (`"It was"` 17k — NOT
caught, no script guard yet) and cross-script dupes (`"Abu Huraira"` Latin vs `أبو هريرة`).
Related: [[project_narrators_two_producers]], [[project_chain_hollow_reads_staging]].
