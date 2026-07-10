# Prod graph reload runbook — #723 data-quality fix

Replaces the polluted/broken prod isnad graph (768k nodes, 8.98% linked, hollow
chains, أبيه pollution) with the corrected graph already validated on stage
(1.67M nodes, ~69% linked, 100%-populated chains, zero name pollution).

**This is a destructive, owner-sign-off-only operation.** Do not run any step in
§3+ without explicit owner approval.

---

## 0. TARGET VERIFICATION — read this every time, before every destructive step

| | Host alias | IP | Role |
|---|---|---|---|
| ✅ TARGET | `noorinalabs-prod` | **178.156.214.225** | production |
| ⛔ NOT this one | `noorinalabs-stg` | 87.99.137.225 | staging (already reloaded) |

Before any wipe/load command, confirm the host:

```bash
ssh noorinalabs-prod 'hostname; curl -s ifconfig.me; echo'
# MUST print the prod hostname and 178.156.214.225. If it prints 87.99.137.225, STOP.
```

Every `ssh`/`SSH_HOST` in this runbook is `noorinalabs-prod`. There is no step
that should ever name the stage host.

---

## 1. Preconditions (all must be true before starting)

1. **deploy #505 merged AND applied to the prod host.** Prod Neo4j must have the
   raised heap/pagecache limits and the host must have active swap — without this
   the dense `PARALLEL_OF` edge phase OOMs (the failure we hit on stage before the
   RAM bump). Verify on the host:
   ```bash
   ssh noorinalabs-prod 'free -h; swapon --show'                 # swap present + sized
   ssh noorinalabs-prod 'docker inspect noorinalabs-neo4j-1 --format "{{.HostConfig.Memory}}"'  # mem limit applied
   ```
2. **Corrected curated artifacts present locally** (the scrubbed set the stage
   reload used): `data/curated/narrators_canonical.parquet` = **232,766** rows,
   `data/curated/narrator_mentions_resolved.parquet` = **3,126,954** rows. Verify:
   ```bash
   python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/curated/narrators_canonical.parquet').num_rows)"
   ```
   > **Regenerating these on the build host?** Before committing to a full
   > multi-hour `resolve`, sanity-check a code change and its throughput against
   > the real data with a bounded probe — `resolve --from-step <stage> --no-resume
   > --stop-after N`. See [Testing changes against a data subset](testing-on-subsets.md)
   > (note the head-biased-sample caveat before trusting a head-of-data rate).
3. **Stage validation is green** (already done): chains 100% populated, top
   narrator = أبو هريرة/سفيان (not أبيه), 0 relational + 0 English-fragment
   residue. This runbook applies the *same* artifacts/image to prod.
4. **Maintenance note**: the prod API reads Neo4j live. During the wipe→reload
   window the prod UI will show an emptying/partial graph. Decide whether to put
   up a maintenance notice first (optional — the load converges in ~1 hr).

---

## 2. Backup current prod graph (safety net)

Even though the current prod graph is the broken one, dump it before wiping so the
operation is reversible:

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

## 3. Wipe prod graph  ⚠️ OWNER SIGN-OFF REQUIRED

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

## 4. Reload from corrected artifacts

Same loader the stage reload used; only the target host differs.

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

An `rc=4` prints `LOAD SUCCEEDED (… nodes, … edges written)` first. Below is
the historical reading of the old conflated `rc=1`; confirm via the Load Summary
line: `total_nodes≈1,665,760`, `total_errors=0`. The 4 flags and why they're benign:
- `chain_integrity: 100 cycle(s)` — **this number was a cap, not a measurement**
  (the query ended in `LIMIT 100` and the classifier counted rows). Fixed in
  da#250: the check now reports exact counts — `0 self-loops; 23,139 reciprocal
  pair(s)` — and only self-loops gate. The reciprocal pairs are an upstream
  over-merge metric, tracked da#248, to be re-measured after da#356.
- `graph_integrity_deferred_inventory` / `sanadset_orphan_inventory: query
  execution failed` — validation harness runs multi-statement `.cypher` as one
  query; tracked da#249.
- `orphan_narrators: ~72,852` — expected bio-dictionary narrators with no mention.

---

## 5. Validate prod (must match stage)

```bash
ssh noorinalabs-prod 'cid=$(docker ps -qf name=neo4j | head -1)
PW=$(docker exec "$cid" sh -c "printenv NEO4J_AUTH" | cut -d/ -f2)
run() { docker exec "$cid" cypher-shell -u neo4j -p "$PW" --format plain "$1"; }
run "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) ORDER BY count(*) DESC"
run "MATCH (c:Chain) RETURN count(*) AS total, sum(CASE WHEN c.narrator_ids IS NULL OR size(c.narrator_ids)=0 THEN 1 ELSE 0 END) AS hollow"
run "MATCH (n:Narrator) WITH count(n) AS t, sum(CASE WHEN COUNT{(n)--()}>0 THEN 1 ELSE 0 END) AS linked RETURN t, linked, round(linked*10000.0/t)/100 AS pct"
run "MATCH (n:Narrator)<-[:NARRATED]-() RETURN n.name_ar AS name, count(*) AS deg ORDER BY deg DESC LIMIT 10"
run "MATCH (n:Narrator) WHERE n.name_ar_normalized IN [\"ابيه\",\"جده\",\"امه\",\"عنه\"] RETURN n.name_ar_normalized, n.mention_count"
'
```

**Pass criteria** (parity with stage):
- Narrators ≈ 232,766; Chains ≈ 585,129 with **hollow = 0**.
- Linkage ≈ 69% (vs 8.98% before).
- Top narrators are real (سفيان/شعبة/أبو هريرة/الزهري/مالك …) — **أبيه absent**.
- Relational-pronoun query returns **0 rows**.

Also confirm the prod app serves: `curl -s -o /dev/null -w '%{http_code}\n'
https://isnad.<prod-domain>` returns 200.

---

## 6. Rollback

If validation fails irrecoverably: the loaders are MERGE-idempotent, so a re-run
converges — first try re-running §4 (it heals partial loads). If the graph must be
reverted to the prior state, restore the §2 dump. Because the corrected artifacts
are the source of truth, "roll forward by re-running the load" is the primary
recovery, not a restore.

---

## Sign-off

- [ ] §0 target verified (prod / 178.156.214.225)
- [ ] §1 preconditions met (#505 applied, artifacts present)
- [ ] **Owner approval to wipe (§3)** — _______________
- [ ] §5 validation passed
- [ ] Prod app serving 200
