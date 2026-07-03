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
- **cluster (fuzzy_cluster)** — the multi-day block-scoring pass (NOT sub-minute — PR1 mislabeled it exempt; corrected). Checkpoints `_UnionFind._parent` + applied-block-index set + scored/merged counters in `cluster_records` (opt-in via `ckpt_dir`; pure callers `cluster_assignment`/quality-harness pass None). **Correctness is EASIER than the streaming stages**: final clusters = connected components = union-order-invariant, and `uf.union` idempotent → resume re-applies only not-yet-applied blocks, result identical. Cadence keyed to SCORED-PAIRS (block cost varies orders of magnitude), not block count. Threaded via `cluster_canonical_narrators(staging_dir=, resume=, stop_after=)`. Added to `RESUMABLE_STEPS`.
- **ner** — EXEMPT (cold-by-design: re-mints uuid4 mention_ids). Its "resume" is reusing the whole mentions file via `resolve --from-step disambiguate`.
- **bio_promote / date stages** — EXEMPT (sub-minute, idempotent re-run).

**CLI:** uniform `resolve --no-resume` → `run_all(resume=False)` → threads to disambiguate/cluster/dedup/parallels. Orthogonal to `--from-step` (which step to start at). See [[project_narrators_two_producers]] for the ordering `--from-step` must preserve.

**da#276 `--stop-after N`** (bounded partial-run probe, sibling of `--no-resume`): built once in the cadence layer via `CheckpointController` (counts checkpoint writes, honors `stop_after`) + `StopAfterReached`. Key design: `StopAfterReached` subclasses **BaseException** (NOT Exception) so it sails through run_all's per-step `except Exception` → pipeline HALTS (no downstream consumes partial output) → CLI maps to `EXIT_STOPPED_AT_LIMIT=3` (0=done, 2=argparse, 1=crash). On stop: checkpoint LEFT on disk (composes with resume), final output NOT written, `<stage>_stopped_at_limit` + perf summary. `RESUMABLE_STEPS={disambiguate,dedup,parallels}`; `--stop-after` with `--from-step` on an exempt stage is a hard argparse error. Cold probe = `resolve --from-step <stage> --no-resume --stop-after N`; head-of-data rate is head-biased (trust for A/B same-prefix, not full-run ETA — fuzzy_cluster early blocks faster, da#270). Docs: `docs/testing-on-subsets.md`.
