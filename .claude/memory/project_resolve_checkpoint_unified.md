---
name: project_resolve_checkpoint_unified
description: da#272 unified resolve crash-resume — shared src/resolve/_checkpoint.py primitives + which stages resume/are exempt + the CLI --no-resume switch.
metadata:
  type: project
---

da#272 extended [[project_disambiguate_checkpoint_resume]]'s pattern to every long resolve stage under ONE shared module: `src/resolve/_checkpoint.py`.

**Shared primitives** (the 7 conventions live here, stages own only their state SHAPE + fingerprint column set):
- `checkpoint_dir(staging, stage)` → `<staging>/.<stage>_checkpoint/` (gitignored).
- `save_checkpoint` (atomic `.tmp`→`os.replace`) / `load_checkpoint` (torn/corrupt → None) / `clear_checkpoint`.
- `resolve_cadence(override, ENV_VAR, default)` — kwarg > env > default, floored at 1, bad env warns+falls back.
- `hash_parquet_column_groups(paths, {group: cols})` — single-pass streaming SHA-256, one digest per column-group; a group's digest is byte-identical to hashing that group alone (so disambiguate's content/mention_id split stayed byte-identical → **pre-da#272 checkpoints still resume**). `hash_strings(*parts)` for non-parquet inputs (dedup).
- `log_resume(stage, skipped=, total=)` — uniform `<stage>_resume` line.

**Coverage after da#272:**
- **disambiguate** — refactored to consume the shared helper (behavior unchanged; existing tests retargeted to `_checkpoint.*`).
- **dedup** — the FAISS search + pair-collection phase (AFTER the da#245 encode memmap) now checkpoints per query row-block. Split into `dedup._search_and_collect_resumable` (index injected via `build_index`/`reload_index` callables) so it is unit-testable without faiss/sentence-transformers. **On resume the persisted `.faiss` index is RELOADED, not rebuilt** — an IVF index's random kmeans init would otherwise diverge the resumed search.
- **parallels** — the anchor scan checkpoints accumulated links + next anchor per block. `indexed`/`postings` rebuilt deterministically; int-keyed partner sets → deterministic order → byte-identical resume.
- **ner** — EXEMPT (cold-by-design: re-mints uuid4 mention_ids). Its "resume" is reusing the whole mentions file via `resolve --from-step disambiguate`.
- **bio_promote / cluster / date stages** — EXEMPT (sub-minute, idempotent re-run).

**CLI:** uniform `resolve --no-resume` → `run_all(resume=False)` → threads to disambiguate/dedup/parallels `run(resume=False)`. Orthogonal to `--from-step` (which step to start at). fuzzy_cluster checkpoint deferred to a follow-up PR (sequenced AFTER da#270's `cluster_records` restructure). See [[project_narrators_two_producers]] for the ordering `--from-step` must preserve.
