# Testing changes against a data subset (`resolve --stop-after`)

A resolve stage can take hours to days on the full corpus. `resolve --stop-after N`
lets you exercise a stage against the **real** data — real inputs, real block-size
distribution, real rates — for a bounded amount of work, then stop cleanly. Use it
to sanity-check a code change and to get a realistic throughput number without
paying for a full run. It is a sibling of `--no-resume` and rides the same
[unified checkpoint machinery](../src/resolve/_checkpoint.py) (da#272 / da#276), so
every resumable stage supports it identically.

## What it does

`--stop-after N` stops the stage cleanly **after its Nth checkpoint write**, where
a checkpoint = `N` batches/blocks at the stage's cadence (counting checkpoints, not
rows, maps 1:1 onto the machinery and is stage-appropriately sized). On stop:

- the **checkpoint is left on disk** — a later bare `resolve --from-step <stage>`
  resumes from it, so a bounded probe composes with a real run;
- the stage's **final output is NOT written** and the **pipeline halts** — no
  downstream stage ever consumes a partial output;
- a compact **perf summary** is printed and logged
  (`<stage>_stopped_at_limit checkpoints=N processed=… rate_per_s=…`) — the number
  you ran the flag to get;
- the process exits with a **distinct status (3)** so a script can tell
  "stopped at limit" from "completed" (0), an argument error (2), or a crash (1).

Only the **resumable** stages honour it: `disambiguate`, `dedup` (its FAISS
search + pair-collection phase), and `parallels` (its anchor scan). The exempt
stages have no checkpoint — `ner` is cold-by-design (it re-mints uuid4
`mention_id`s each run) and `bio_promote` / `cluster` / the date stages are
sub-minute idempotent re-runs — so pairing `--stop-after` with `--from-step` on an
exempt stage is a hard error, not a silent no-op.

## The cold bounded probe

To measure one stage from a clean start against the real data:

```bash
# Probe the first 5 checkpoints of dedup's FAISS/collection phase, cold.
resolve --from-step dedup --no-resume --stop-after 5
```

- `--from-step dedup` skips the earlier stages (their outputs must already be on
  disk) and starts at `dedup`.
- `--no-resume` discards any existing `dedup` checkpoint so the probe starts cold
  (a fresh, comparable sample every time).
- `--stop-after 5` stops after 5 checkpoint writes and prints the perf summary.

The three flags compose freely. Drop `--from-step` to bound the first resumable
stage (`disambiguate`); drop `--no-resume` to probe *continued* work from an
existing checkpoint.

## Baseline-vs-change comparison

To decide whether a change helped or hurt throughput, run the **same** cold
bounded probe on both revisions and compare the `rate_per_s` in the perf summary:

```bash
git checkout main            && resolve --from-step dedup --no-resume --stop-after 5   # baseline rate
git checkout my-change       && resolve --from-step dedup --no-resume --stop-after 5   # candidate rate
```

Because both probes cover the identical head-of-data prefix (same inputs, same
first N blocks), the rate difference isolates your change. Use `--no-resume` on
both so neither silently continues a stale checkpoint.

## Caveat: the first N batches are a head-biased sample

A bounded probe measures the **head** of the data, which is not always
representative of the whole run:

- **Rates can drift with position.** `fuzzy_cluster`, for example, scores its
  early blocks faster than its late ones (the throughput degradation tracked in
  da#270), so a head-of-data rate over-estimates full-run throughput. `dedup`'s
  per-block rate is comparatively flat, but its FAISS index build is a one-time
  cost paid before the first block, so a tiny `N` amortises it poorly.
- **Composition can drift with position.** If the corpus is ordered (by source,
  by ingest time, …), the first N blocks may be dominated by one source and miss
  the cross-source cases a later block hits.

So: trust a head-of-data rate for an **A/B comparison of the same prefix** (the
bias cancels), and for catching a gross regression or crash early. Do **not**
extrapolate a head rate to a full-run ETA for a stage whose rate is known to drift
with position — take a larger `N`, or probe a few separate offsets, before
trusting the number as a whole-run estimate.

## See also

- [`src/resolve/_checkpoint.py`](../src/resolve/_checkpoint.py) — the shared
  cadence + checkpoint layer that implements `--no-resume` and `--stop-after`.
- [Prod reload runbook](PROD_RELOAD_RUNBOOK_723.md) — regenerating the curated
  artifacts on the build host before a reload.
