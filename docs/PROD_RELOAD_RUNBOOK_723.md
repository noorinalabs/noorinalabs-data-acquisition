# Prod graph reload runbook — #723 data-quality fix

> **STATUS (measured 2026-08-09, da#506): the #723 reload has ALREADY been applied to prod.**
> Prod carries the corrected graph — 2,722,526 nodes, 102,350 Narrators, **hollow = 0**, zero
> relational-pronoun pollution, and the same `LoadProvenance.parquet_ref` as stage
> (`staged/narrator-resolve/2026-07-25-c5a95be`). **The destructive wipe in §4 is NOT the
> default path and must not be run to "apply #723" — that work is done.** The only outstanding
> gap is a small edge-phase shortfall vs stage (§2), healed by the **idempotent top-up in §3**
> with no wipe. §4 exists solely as a contingency for a graph that is genuinely broken, and it
> is reachable only after §1 says so.

Historical purpose: replace the polluted/broken prod isnad graph (768k nodes, 8.98% linked,
hollow chains, أبيه pollution) with the corrected graph validated on stage. That replacement
has happened. What remains is parity maintenance, not replacement.

**§4 is a destructive, owner-sign-off-only operation.** Do not run any step in §4 without
explicit owner approval, and not at all unless §1 classifies prod as BROKEN.

---

## 0. TARGET VERIFICATION — read this every time, before every destructive step

| | Host alias | IP | Role |
|---|---|---|---|
| ✅ TARGET | `noorinalabs-prod` | **178.156.214.225** | production |
| ⛔ NOT this one | `noorinalabs-stg` | 87.99.137.225 | staging |

Before any wipe/load command, confirm the host:

```bash
ssh noorinalabs-prod 'hostname; curl -4 -s ifconfig.me; echo'
# MUST print the prod hostname and 178.156.214.225. If it prints 87.99.137.225, STOP.
```

> **`-4` is load-bearing — do not drop it.** Both hosts are dual-stack and `curl -s ifconfig.me`
> without it returns the host's **IPv6** address (`2a01:4ff:f4:a690::1` on prod,
> `2a01:4ff:f4:5e0d::1` on stage), which matches *neither* IPv4 literal in the table above. The
> unflagged form therefore never confirms and never rejects — it fails identically on the right
> host and the wrong one, which is the same as having no guard at all in front of an
> irreversible `DETACH DELETE`. Verified on both hosts 2026-08-09 (da#506, defect 4).

Every `ssh`/`SSH_HOST` in this runbook is `noorinalabs-prod`. There is no step
that should ever name the stage host.

---

## 1. Measure before you decide — classify the current prod graph

**Run this first, every time.** Nothing below may be chosen from memory or from this
document's prose: the previous revision of this runbook asserted a prod state that was
~1 month out of date, and following it would have wiped a healthy graph (da#506).

```bash
ssh noorinalabs-prod 'cid=$(docker ps -qf name=neo4j | head -1)
PW=$(docker exec "$cid" sh -c "printenv NEO4J_AUTH" | cut -d/ -f2)
run() { docker exec "$cid" cypher-shell -u neo4j -p "$PW" --format plain "$1"; }
run "MATCH (n) RETURN count(n) AS total_nodes"
run "MATCH (c:Chain) RETURN count(*) AS chains, sum(CASE WHEN c.narrator_ids IS NULL OR size(c.narrator_ids)=0 THEN 1 ELSE 0 END) AS hollow"
run "MATCH (n:Narrator) WITH count(n) AS t, sum(CASE WHEN COUNT{(n)--()}>0 THEN 1 ELSE 0 END) AS linked RETURN t, linked, round(linked*10000.0/t)/100 AS pct"
run "MATCH (n:Narrator) WHERE n.name_ar_normalized IN [\"ابيه\",\"جده\",\"امه\",\"عنه\"] RETURN n.name_ar_normalized, n.mention_count"
run "MATCH (p:LoadProvenance) RETURN p.parquet_ref"
'
```

Classify on the result:

| Observation | Verdict | Go to |
|---|---|---|
| `hollow = 0`, linkage ≈ 59–60%, pronoun query returns 0 rows, `parquet_ref` = `staged/narrator-resolve/2026-07-25-c5a95be` | **HEALTHY** — #723 already applied | §2 (compare to stage), then §3 only if short |
| `hollow > 0`, or linkage ≈ 9%, or أبيه present as a narrator | **BROKEN** | §5 backup → §4 wipe → §6 reload (owner sign-off) |
| anything else | **UNCLASSIFIED** — stop and escalate | — |

> The pronoun query is a **controlled** check: its query shape does return rows for names that
> do exist in the graph, so an empty result is a real zero and not a silently-broken query.

**As measured 2026-08-09, prod classifies HEALTHY.**

### 1a. Preconditions (only if you will run §4/§6)

1. **deploy #505 merged AND applied to the prod host.** Prod Neo4j must have the
   raised heap/pagecache limits and the host must have active swap — without this
   the dense `PARALLEL_OF` edge phase OOMs (the failure we hit on stage before the
   RAM bump). Verified applied 2026-08-02: 8 GiB swap active, Neo4j mem limit 10 GiB. Re-check:
   ```bash
   ssh noorinalabs-prod 'free -h; swapon --show'                 # swap present + sized
   ssh noorinalabs-prod 'docker inspect noorinalabs-neo4j-1 --format "{{.HostConfig.Memory}}"'  # mem limit applied
   ```
2. **Corrected curated artifacts present locally** — verify by reading the files, and
   compare against **stage**, never against a number typed into this document:
   ```bash
   python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/curated/narrators_canonical.parquet').num_rows)"
   python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/curated/narrator_mentions_resolved.parquet').num_rows)"
   ```
   Measured 2026-08-09: `narrators_canonical` = **102,350**, `narrator_mentions_resolved` =
   **2,975,629**. The canonical row count must equal the `Narrator` count on stage (102,350) —
   that equality, not a literal in this file, is the real precondition.

   > **Do not "correct" a mismatch against an older number.** These counts moved legitimately:
   > narrators *fell* (232,766 → 102,350) via the merge fixes da#347/#356/#376, and chains
   > *rose* via the corpus recovery da#364–369. An earlier revision of this runbook pinned the
   > pre-fix figures and would have aborted a correct reload (da#506, defect 1).

   > **Regenerating these on the build host?** Before committing to a full
   > multi-hour `resolve`, sanity-check a code change and its throughput against
   > the real data with a bounded probe — `resolve --from-step <stage> --no-resume
   > --stop-after N`. See [Testing changes against a data subset](testing-on-subsets.md)
   > (note the head-biased-sample caveat before trusting a head-of-data rate).
3. **Stage validation is green.** Confirm against stage live (§2), not from prose.
4. **Maintenance note**: the prod API reads Neo4j live. §3 is additive and needs no notice.
   A §4 wipe→reload window shows an emptying/partial graph in the prod UI — put up a
   maintenance notice first (the load converges in ~1 hr).

---

## 2. Compare prod against stage (parity check)

Run the §1 block against `noorinalabs-stg` as well and diff. Measured 2026-08-09:

| | stage | prod | delta |
|---|---|---|---|
| total nodes | 2,722,570 | 2,722,526 | −44 |
| Hadith | 1,427,222 | 1,427,222 | — |
| Chain | 1,122,521 | 1,122,477 | −44 |
| Narrator | 102,350 | 102,350 | — |
| Grading | 69,465 | 69,465 | — |
| Collection | 1,011 | 1,011 | — |
| hollow chains | 0 | 0 | — |
| linked Narrators | 61,278 (59.87%) | 61,212 (59.81%) | −66 |
| `PARALLEL_OF` | 7,549,479 | 7,549,479 | — |
| `TRANSMITTED_TO` | 3,387,179 | 3,342,894 | −44,285 |
| `APPEARS_IN` | 1,426,899 | 1,426,899 | — |
| `NARRATED` | 839,238 | 826,376 | −12,862 |
| `STUDIED_UNDER` | 55,248 | 55,248 | — |
| `GRADED_BY` | 48,283 | 48,283 | — |

Nodes and the bulk edge phases match. The shortfall is confined to the **chain-derived**
phases (`Chain`, `TRANSMITTED_TO`, `NARRATED`) — a truncated tail on the prod run, **not**
pollution and **not** a hollow-chain defect. A shortfall of this shape is a §3 top-up.

---

## 3. Top-up: idempotent re-load, no wipe  ← DEFAULT PATH when §1 says HEALTHY

The loaders are `MERGE`-idempotent, so re-running the load against a healthy-but-short graph
converges it to parity without deleting anything. This is the runbook's own documented
"roll forward" recovery, promoted to the primary path.

```bash
cd noorinalabs-data-acquisition
SSH_HOST=noorinalabs-prod LOAD_ARGS="load" scripts/load_staging.sh
```

Then re-run §2 and confirm the deltas have closed. **Non-destructive**, but it does write to
prod — keep it owner-gated. If §2 still shows a shortfall after two top-ups, escalate rather
than reaching for §4: a load that cannot converge is a loader defect, and wiping hides it.

---

## 4. Wipe prod graph  ⚠️ OWNER SIGN-OFF REQUIRED — CONTINGENCY ONLY

**Do not run this to apply #723 — #723 is already applied (§1).** Reach this step only when
§1 classifies prod as **BROKEN**, and only with explicit owner approval.

Detached, wipe-until-zero loop (same pattern that cleared stage; survives SSH
drops, handles dense nodes with a smaller final batch). **Re-run §0 target check
immediately before this.**

```bash
ssh noorinalabs-prod '
  cid=$(docker ps -qf name=neo4j | head -1)
  PW=$(docker exec "$cid" sh -c "printenv NEO4J_AUTH" | cut -d/ -f2)
  nohup sh -c "
    while true; do
      n=\$(docker exec '"$cid"' cypher-shell -u neo4j -p '"$PW"' --format plain \
        \"MATCH (n) RETURN count(n)\" | tail -1 | tr -dc 0-9)
      echo \"\$(date -u +%H:%M:%SZ) nodes=\$n\"
      [ \"\$n\" = 0 ] && echo WIPE COMPLETE && break
      bs=5000; [ \"\$n\" -lt 50000 ] && bs=1000
      docker exec '"$cid"' cypher-shell -u neo4j -p '"$PW"' \
        \"CALL apoc.periodic.iterate('"'"'MATCH (n) RETURN n'"'"','"'"'DETACH DELETE n'"'"',{batchSize:\$bs,parallel:false})\" >/dev/null 2>&1
    done
  " > /tmp/prod-wipe.log 2>&1 &
  echo "wipe started, watch /tmp/prod-wipe.log"
'
# Poll: ssh noorinalabs-prod 'tail -3 /tmp/prod-wipe.log'  — wait for "WIPE COMPLETE"
```

---

## 5. Backup current prod graph (safety net — run BEFORE §4)

Dump the graph before wiping so the operation is reversible:

```bash
ssh noorinalabs-prod '
  cid=$(docker ps -qf name=neo4j | head -1)
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  docker exec "$cid" neo4j-admin database dump neo4j --to-path=/backups 2>&1 | tail -5 || \
    echo "NOTE: online dump may require neo4j stop; the reload is itself reproducible from artifacts, so a dump failure is non-fatal."
  echo "backup tag: $ts"
'
```

The reload is fully reproducible from the curated artifacts + loader image, so the
dump is a belt-and-suspenders measure, not the primary recovery path.

---

## 6. Full reload from corrected artifacts (after a §4 wipe)

Same loader the stage reload used; only the target host differs. Mechanically identical to
§3 — the difference is that §3 converges an existing graph while this rebuilds an empty one.

```bash
cd noorinalabs-data-acquisition
SSH_HOST=noorinalabs-prod LOAD_ARGS="load" scripts/load_staging.sh
```

This: rsyncs code + node-bearing staging parquet + scrubbed curated artifacts to
prod, builds the loader image on the prod host, runs the loader **detached** on
the `noorinalabs_backend` network, and short-polls to completion (~1 hr, the
`PARALLEL_OF` edge phase dominates).

**Historical `rc=1` false alarm — fixed by da#354.** `load` used to exit `1`
both when the load genuinely failed and when the load succeeded but post-load
validation reported findings, so `rc=1` was indistinguishable from a real
failure. The exit codes are now separate:

| rc | meaning |
|----|---------|
| `0` | load succeeded, validation clean (warnings allowed) |
| `1` | **the load failed** — the graph was not fully written |
| `4` | **the load succeeded**, validation reported findings |

An `rc=4` prints `LOAD SUCCEEDED (… nodes, … edges written)` first. Confirm via the Load
Summary line: `total_nodes≈2,722,570` (stage parity) and `total_errors=0`. The 4 flags and
why they're benign:
- `chain_integrity: 100 cycle(s)` — **this number was a cap, not a measurement**
  (the query ended in `LIMIT 100` and the classifier counted rows). Fixed in
  da#250: the check now reports exact counts — `0 self-loops; 23,139 reciprocal
  pair(s)` — and only self-loops gate. The reciprocal pairs are an upstream
  over-merge metric, tracked da#248, to be re-measured after da#356.
- `graph_integrity_deferred_inventory` / `sanadset_orphan_inventory: query
  execution failed` — validation harness runs multi-statement `.cypher` as one
  query; tracked da#249.
- `orphan_narrators: ≈41,072` — expected bio-dictionary narrators with no mention
  (102,350 total − 61,278 linked, stage 2026-08-09). The previously-documented
  `~72,852` was the pre-merge-fix figure.

---

## 7. Validate prod (must match stage)

```bash
ssh noorinalabs-prod 'cid=$(docker ps -qf name=neo4j | head -1)
PW=$(docker exec "$cid" sh -c "printenv NEO4J_AUTH" | cut -d/ -f2)
run() { docker exec "$cid" cypher-shell -u neo4j -p "$PW" --format plain "$1"; }
run "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) ORDER BY count(*) DESC"
run "MATCH (c:Chain) RETURN count(*) AS total, sum(CASE WHEN c.narrator_ids IS NULL OR size(c.narrator_ids)=0 THEN 1 ELSE 0 END) AS hollow"
run "MATCH (n:Narrator) WITH count(n) AS t, sum(CASE WHEN COUNT{(n)--()}>0 THEN 1 ELSE 0 END) AS linked RETURN t, linked, round(linked*10000.0/t)/100 AS pct"
run "MATCH (n:Narrator)-[:NARRATED]->() RETURN n.name_ar AS name, count(*) AS deg ORDER BY deg DESC LIMIT 10"
run "MATCH (n:Narrator) WHERE n.name_ar_normalized IN [\"ابيه\",\"جده\",\"امه\",\"عنه\"] RETURN n.name_ar_normalized, n.mention_count"
'
```

> **The `NARRATED` arrow points OUT of the narrator: `(n:Narrator)-[:NARRATED]->()`.**
> An earlier revision wrote `(n:Narrator)<-[:NARRATED]-()`, which returns **zero rows on a
> perfectly healthy graph** — an operator reads the empty top-narrator list as catastrophic
> data loss and may wipe a good graph in response. Verified 2026-08-09: 839,238 `NARRATED`
> edges on stage, all `Narrator → Hadith` (da#506, defect 3).

**Pass criteria** (parity with stage, measured 2026-08-09 — re-measure stage rather than
trusting these literals if the artifacts have moved):
- Narrators = **102,350**; Chains ≈ **1,122,521** with **hollow = 0**.
- Total nodes ≈ **2,722,570**.
- Linkage ≈ **59.9%** (vs 8.98% on the pre-#723 graph).
- Top narrators are real (ابن عباس، ابن عبس، أبو عبد الله الحافظ، وكيع، سفيان، علي …) —
  **أبيه absent**.
- Relational-pronoun query returns **0 rows**.

Also confirm the prod app serves: `curl -s -o /dev/null -w '%{http_code}\n'
https://isnad.<prod-domain>` returns 200.

---

## 8. Rollback

If validation fails irrecoverably: the loaders are MERGE-idempotent, so a re-run
converges — first try re-running §3/§6 (it heals partial loads). If the graph must be
reverted to the prior state, restore the §5 dump. Because the corrected artifacts
are the source of truth, "roll forward by re-running the load" is the primary
recovery, not a restore.

---

## Sign-off

- [ ] §0 target verified (prod / 178.156.214.225, via `curl -4`)
- [ ] §1 measured and classified (HEALTHY → §3 · BROKEN → §5/§4/§6 · UNCLASSIFIED → escalate)
- [ ] §2 prod-vs-stage parity diffed
- [ ] §1a preconditions met (#505 applied, artifacts match stage) — only if §4/§6
- [ ] **Owner approval to write to prod** — _______________
- [ ] **Owner approval to wipe (§4)** — _______________ _(contingency only; NOT required for §3)_
- [ ] §7 validation passed
- [ ] Prod app serving 200
