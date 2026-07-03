---
name: project_disambiguate_checkpoint_resume
description: da#268 disambiguate crash-resume — mention_id is uuid4 (random per NER run) so resume must reuse the file via --from-step; fingerprint excludes it.
metadata:
  type: project
---

da#268 made `src/resolve/disambiguate.py::run` crash-resumable (checkpoint the 3.1M-mention streaming loop every N batches under `data/staging/.disambiguate_checkpoint/`).

**Load-bearing gotcha:** NER mints `mention_id` via `str(uuid.uuid4())` (`src/resolve/ner.py:85,174`) — it is RANDOM, regenerated every NER run, NOT content-deterministic. mention_id is only an internal join key inside resolve (disambiguate `resolved_map`→backfill, merge_log/ambiguous audit rows); the graph keys off `canonical_narrator_id`, never mention_id (`src/graph/load_*.py`). merge_log/ambiguous are audit-only (no graph/enrich consumer).

**Consequence for resume:** a resumed run can only be OUTPUT-IDENTICAL to a cold run if it reads the SAME mentions file (same mention_ids), because mention_ids are embedded in the output artifacts (merge_log, ambiguous, backfill join). So the sanctioned recovery command is `resolve --from-step disambiguate`, which SKIPS ner and reuses the existing `narrator_mentions_resolved.parquet` verbatim. A bare `resolve` re-run regenerates NER mention_ids → checkpoint correctly cold-starts (safe, not corrupt).

**Fingerprint design:** two SHA-256 sub-hashes stored in checkpoint meta — `content_hash` over stable driving cols (hadith_id, source_corpus, position_in_chain, name_raw, name_normalized, transmission_method) and `mention_id_hash` over the mention_id col; resume requires BOTH to match. content-match + mention_id-mismatch logs a precise "use --from-step disambiguate" hint then cold-starts. Excludes canonical_narrator_id/confidence (rewritten by the end-of-run backfill) and mtime — so an identical-content NER rewrite doesn't spuriously invalidate the corpus check.

Deviation from issue text: the issue's "identical NER rewrite must not invalidate" is only fully achievable if mention_id were deterministic; it isn't, and making it so risks graph-identity blast radius, so I kept NER unchanged and routed recovery through `--from-step disambiguate`. See [[project_narrators_two_producers]] for the run_all ordering the --from-step skip must preserve.
