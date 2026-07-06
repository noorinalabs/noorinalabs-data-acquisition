---
name: feedback_dual_detector_cross_sect_authority
description: Two producers writing one composed artifact must derive every field from the SAME authoritative source; a test green-in-CI/red-locally on an embedder-dependent assertion is a real divergence to hunt, not env flake. da#321.
metadata:
  type: feedback
---

**da#321.** `test_run_all_promoted_bio_only_narrator_survives` passed in CI but failed deterministically in the local `.venv` (`cross_sect: [[false]]`). Not env flake — a **real correctness bug**.

**Why:** `PARALLEL_OF` has two detectors that both write `staging/parallel_links.parquet`, and `run_all._compose_parallel_links` unions them keyed on the canonical `(hadith_id_a, hadith_id_b)` pair, letting the **semantic** (`dedup.py`) row win on collision. But the two derived `cross_sect` from **different sources of truth**:
- `parallels.py` (deterministic lexical): `bool(sect_i and sect_j and sect_i != sect_j)` — the authoritative **`sect` column**.
- `dedup.py` (semantic): `_is_cross_sect(corpus_a, corpus_b)` — inferred from the source-**corpus NAME** via `_SUNNI_SOURCES`/`_SHIA_SOURCES` allowlists.

Same-corpus/different-sect pairs (fixtures both default `source_corpus="sunnah"`, explicit `sect` sunni/shia) → dedup emits `cross_sect=False` and clobbers the correct deterministic `True`. **Locally the MiniLM model loads** so dedup emits the clobbering row; **in CI the model is absent** so dedup degrades to empty and the deterministic `True` survives → the exact CI/local split. Same class as [[project_relational_pollution_scrub_equiv]]-adjacent embedder-env parity (parent `project_semantic_embedder_parity`, deploy#523).

**Fix (PR#322):** align `dedup` on the authoritative `sect` column — `_load_hadith_texts` loads `sect` (non-nullable `HADITH_SCHEMA` field) not `source_corpus`; `_is_cross_sect(sect_a, sect_b)`; `id_to_corpus`→`id_to_sect`; dedup checkpoint `schema_version` 1→2 (it persists computed `cross_sects`, so a v1 checkpoint holds stale corpus-derived values). Also fixed a **latent prod mislabel**: `fawaz.py` sets sect per-edition but all fawaz share one `source_corpus="fawaz"` (was in the sunni allowlist) → cross-edition fawaz pairs wrongly `False`; `bihar`/`halimbahae`/`tusi`/`mis` were in neither allowlist → unconditionally `False`.

**Two reusable heuristics:**
1. **Shared-artifact / compose = shared source of truth.** When N producers write one artifact and a merge step picks a per-key winner, every field must come from the same authoritative column across producers, or the composed value flips with which producer wins.
2. **Green-CI + red-local on an embedder/model-dependent assertion ≠ flake.** It usually means an outcome that varies by whether the model is present — hunt the real divergence (which code path only runs when the model loads), don't paper over the test. Verify a fix by confirming the previously-failing test passes **with the model loaded** (0 skipped), not just that CI stays green.
