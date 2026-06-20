---
name: project_narrators_two_producers
description: "narrators_canonical.parquet has TWO producers (disambiguate overwrites, bio_promote merges) — run_all must order disambiguate BEFORE bio_promote"
metadata: 
  node_type: memory
  type: project
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

As of P4W3, `narrators_canonical.parquet` (the canonical Narrator records the
graph loader ingests) has **two producers**, both keying ids on the single
`identity.make_canonical_id` (da#110, nar:uuid5(CANONICAL_NAMESPACE,
normalized_name) — same scheme da#99 uses, so the same name collapses to one
node across both paths):

- **mention-driven** — `src/resolve/disambiguate.py::run` (da#99 adds a bio-less
  exact-name fallback so ALL named mentions canonicalize). It **OVERWRITES**
  `narrators_canonical.parquet` from its own in-memory map (does not read existing).
- **bio-direct** — `src/resolve/bio_promote.py::promote_bios_to_canonical`
  (da#110, Itqan rijal). It is **MERGE-SAFE**: reads any existing
  `narrators_canonical.parquet` and extends it.

**Invariant:** whoever wires these into the pipeline (`src/resolve/__init__.py`
run_all) MUST order **disambiguate BEFORE bio_promote** — otherwise disambiguate
clobbers the bio-only (profile-source) narrators bio_promote produced. The fix if
disambiguate ever needs to run second: make its write merge-safe too (read-existing
+ extend), mirroring bio_promote.

**RESOLVED da#117 / PR#127 (merged wave-4 2026-06-12, ancestor of wave-6):**
bio_promote IS now wired into `run_all` — order `NER → disambiguate → bio_promote →
(dedup + detect_parallels)`, invariant documented inline, with ordering +
bio-only-survival tests in `tests/test_resolve/test_run_all.py`. The earlier "NOT
yet in run_all / future-wiring hazard" framing is stale. P4W6 #120 re-filed the
same gap and was **closed as a verified dup of #117** (no re-implementation).

Related: [[feedback_appears_in_merge_null]] (da#109 backfills
canonical_narrator_id onto resolved mentions to wire NARRATED edges — separate
follow-on).
