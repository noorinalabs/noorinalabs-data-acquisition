---
name: project_lk_latin_english_isnad_fork
description: da#298 — lk's 42k empty-name_ar mentions are English-isnad romanized duplicates of the parallel Arabic chain; no per-mention Arabic recoverable (99.5% chain-length mismatch). Drop-vs-keep is an owner policy call; _LATIN_FALLBACK_POLICY defaults keep.
metadata:
  type: project
last_verified: 2026-07-20
---

**da#298 investigation (measured on `data/staging/narrator_mentions_lk.parquet`, 229,400 mentions).**

`narrator_mentions_lk.parquet` has **41,998 mentions (18.3%) with empty `name_ar` + populated Latin `name_en`** (e.g. `Mughirah ibn Shu'bah`). `ner._load_phase1_mentions` does `name_ar or name_en`, so those fall back to Latin → `normalize_arabic` leaves them Latin → `make_canonical_id` mints Latin-keyed canonical nodes that never merge cross-script (the fork class fixed for fawaz in da#286).

**Root cause (structural, not a bug):** `lk_corpus.run` extracts narrator mentions from BOTH isnads of each hadith — the Arabic isnad (`name_ar`, `chain_index=0`) AND the parallel English-translation isnad (`name_en` Latin, `chain_index=1`). The 41,998 Latin mentions are ALL from the English path (verified: 100% carry `:en:` mention_id + chain 1). They are romanized **duplicates** of narrators already captured in Arabic on the parallel chain.

**Why the parser cannot recover Arabic (the "preferred fix" is unavailable):**
- Raw lk CSV has **no structured Arabic narrator column** — only free-text `Arabic_Isnad` / `English_Isnad` (NER-extracted separately).
- The English & Arabic isnads are **separate chains with independent positions** (da#282 — mixing them fabricates cross-language adjacencies). Empirically: of 33,469 hadiths with an English-isnad mention, **100% also have an Arabic-isnad mention**, but the two chains have **different lengths in 99.5%** of hadiths (English systematically shorter, delta −3 to −6). So a per-mention Arabic name is NOT recoverable by positional alignment; recovery would need cross-lingual NER alignment / transliteration — the lossy path the fawaz note in `ner.py` explicitly rejects.

**Mechanism shipped (da#298):** `ner._LATIN_FALLBACK_POLICY` gate in `_load_phase1_mentions`. `"keep"` (default) = status quo, no data loss. `"drop"` = skip Latin-only cross-script fallbacks (empty `name_ar` + non-Arabic fallback via `is_arabic`). Dropping ~42k mostly-duplicative Latin edges is **owner-visible** → default stays `keep`; the drop decision was surfaced to the owner, not taken unilaterally.

Related: [[project_canonical_identity_invariant]] (id+name = f(mention)), da#286 fawaz→Arabic, da#282 en/ar chain separation.
