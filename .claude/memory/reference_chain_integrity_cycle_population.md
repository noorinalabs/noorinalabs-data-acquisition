---
name: reference_chain_integrity_cycle_population
description: TRANSMITTED_TO cycle population measured on stg — 0 self-loops, 23,139 reciprocal pairs over 5,803 narrators; how to count it so the query terminates.
metadata:
  type: reference
---

Measured on stg 2026-07-09 (`160,614` Narrator, `2,679,527` TRANSMITTED_TO), da#250/da#248.

| quantity | value |
|---|---|
| self-loops (cycle length 1) | **0** |
| reciprocal pairs (length 2), DISTINCT | **23,139** |
| narrators in a reciprocal pair | **5,803** |
| reciprocal *edge instances* | 295,576 |

**1. `count(*)` is not the pair count.** `MATCH (a)-[:TRANSMITTED_TO]->(b) WHERE a.id < b.id AND EXISTS { MATCH (b)-[:TRANSMITTED_TO]->(a) } RETURN count(*)` yields **295,576** — one row per *edge*, and there is one TRANSMITTED_TO per hadith between the same pair. The pair count needs `count(DISTINCT [a.id, b.id])` → 23,139. Reporting 295,576 as "pairs" is a 12.8× overcount.

**2. Variable-length cycle enumeration does not terminate here.** `MATCH path = (n:Narrator)-[:TRANSMITTED_TO*1..20]->(n)` never returns; **even `*1..2` was killed at 150s having emitted zero rows**. Read from stdout alone that is indistinguishable from "0 cycles" — always attach a terminal `printf 'QUERY_RC=%s\n' $?` sentinel and treat its **absence** as "did not complete." (`timeout` returns 124.) See [[feedback_silent_zero_is_not_a_measurement]] in the org memory.

**3. The terminating formulation** (~109s at full stg scale, one statement, no LIMIT) is what `queries/validation/chain_integrity.cypher` now ships:

```cypher
CALL { MATCH (n:Narrator)-[:TRANSMITTED_TO]->(n) RETURN count(DISTINCT n.id) AS self_loops }
CALL { MATCH (a:Narrator)-[:TRANSMITTED_TO]->(b:Narrator)
       WHERE a.id < b.id AND EXISTS { MATCH (b)-[:TRANSMITTED_TO]->(a) }
       RETURN count(DISTINCT [a.id, b.id]) AS reciprocal_pairs }
RETURN self_loops, reciprocal_pairs
```

Two `CALL {}` aggregation subqueries each return exactly one row even on empty input, so the composed result is always one row. A single `MATCH … WITH count(…) … MATCH …` chain would return **zero** rows when the second MATCH finds nothing (grouped aggregation over an empty input), which the classifier would then have to treat as a non-answer.

**4. Cycles of length ≥ 3 are NOT measured, by design.** Enumerating them is the non-terminating expansion above. The shipped gate counts lengths 1 and 2 exactly and *says so* in its `details` string. Before da#250 the gate ended in `LIMIT 100` and the classifier counted rows, so it reported the constant `100 cycle(s) detected` forever — a cap presented as a measurement, and the rc=1 that main#723 misread as a data defect.

**5. `reciprocal_pairs` is a METRIC, not a loader defect.** "A taught B and B taught A" is historically impossible, so every pair is manufactured upstream by identity collapse in `resolve` (da#356). It is a thermometer for over-merge: **re-measure after da#356 lands**; do not delete cycles. The head of the offender ranking (`queries/analysis/chain_integrity_mononym_ranking.cypher`, also rewritten to terminate) is all bare mononyms — عبد الله (829 partners), سفيان (422), محمد (419), شعبة (360), الحسن (304) — corroborating [[project_relational_pollution_scrub_equiv]] and `src/resolve/mononym_split.py`.

Only `self_loops` gates (0 on stg+prod today). Related: [[reference_graph_ops_cypher_shell]] (misspelled property returns NULL for every row and never errors — enumerate `keys(n)` first).
