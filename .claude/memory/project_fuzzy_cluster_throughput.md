---
name: project_fuzzy_cluster_throughput
description: da#270 fuzzy_cluster 3x throughput decay — cdist O(K²) over flattened match-keys is ~99% of cost; a pathological tail (mega-alias + pollution-name records) inflates K and block count. Fix = per-record key + blocking-token caps.
metadata:
  type: project
---

da#270: `src/resolve/fuzzy_cluster.cluster_records` throughput decayed ~3× over the wave-23 run (1,957→600 pairs/s) while CPU *rose* (625%→1090%). Rising CPU ruled OUT the serial `_apply`/union-find and the drain logic (a serial bottleneck would starve workers and drop CPU) — the growing cost is in the **workers**.

**Profiled on the real ~210k-canonical snapshot:** the per-block `process.cdist(flat_keys, flat_keys, token_set_ratio)` is **~99% of wall time** (the Python `_stateless_merge_ok`/`_token_order_consistent` guard-loop is ~1%). cdist is **O(K²) in the flattened match-key count** K = Σ(name+aliases) over the block's members (~0.85 µs/cell). Match-keys/record: p99=19 but **max=740**; significant-tokens/record: p99=34 but **max=1213**. A pathological tail — a few prolific narrators that accreted hundreds of real alias spellings, PLUS pollution nodes whose "name" is a captured isnad fragment (`[ 103 ] - ابراهيم...`, a single 141-token "name", mc=0) — blows up both per-block K (cdist) and the block *count* (a 1213-token record joins C(1213,2)≈735k composite blocks → the multi-GB pair-index build blowup). These records are appended late (bio_promote rijal), so later blocks carry the heavy K → the decay.

**Fix (this PR):** two per-record caps in `fuzzy_cluster.py`, both default 64, touching <1% of records (never a normal name):
- `_MAX_MATCH_KEYS_PER_RECORD` — truncate `_match_keys` (name always first). Bounds per-block cdist K. Measured **3.45× cdist speedup on the worst real block** at cap 64; only lost merges are those justified solely by a record's 65th+ alias.
- `_MAX_BLOCKING_TOKENS_PER_RECORD` — cap significant tokens used to form composite blocking pairs in `cluster_records`. Bounds the C(t,2) block-count explosion + build memory; guards still read the full token set.

Both `None`-disable to exact pre-da#270 behaviour (the equivalence/precision-recall harness path). NOT a merge-semantics change for normal records. See [[project_relational_pollution_scrub_equiv]] (da#247) — the pollution-name records are the same NER residue. da#272 (Alejandra) adds cluster checkpoint/resume AFTER this lands.
