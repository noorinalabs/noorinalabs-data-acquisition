// da#250 — TRANSMITTED_TO cycle census. Returns EXACTLY ONE aggregate row.
//
// There is deliberately no LIMIT. A gate that caps its own result reports the
// cap, not the population. The previous form was:
//
//     MATCH path = (n:Narrator)-[:TRANSMITTED_TO*1..20]->(m:Narrator)
//     WHERE n = m
//     RETURN n.id AS narrator_id, length(path) AS cycle_length
//     LIMIT 100
//
// paired with a classifier that counted returned rows. On the full graph it
// therefore reported "100 cycle(s) detected" — a constant, unchanged by anything
// that happened to the graph, which read as reassurance. Measured against stg on
// 2026-07-09 (160,614 Narrator, 2,679,527 TRANSMITTED_TO) the true population was
// 0 self-loops and 23,139 reciprocal pairs spanning 5,803 narrators. The old form
// also PASSED on an empty result: an aggregate query that returns no row did not
// run, and a zero nobody measured is not a zero.
//
// COVERAGE — read this before trusting a PASS:
//   self_loops        cycle length 1   (n)-[:TRANSMITTED_TO]->(n)          EXACT
//   reciprocal_pairs  cycle length 2   (a)->(b) and (b)->(a), a.id < b.id  EXACT
//   cycles of length >= 3                                                  NOT MEASURED
//
// Enumerating cycles of length >= 3 means the variable-length expansion above,
// which does not terminate at this scale — even a *1..2 bound was killed at 150s
// having emitted no rows at all. Rather than cap it and present the cap as a
// finding, this file measures the two tractable classes exactly and states the
// gap out loud.
//
// GATE vs METRIC:
//   * self_loops is the GATE. A narrator transmitting to himself is
//     unambiguously corrupt. Currently 0 on stg and prod, so the guard is live.
//   * reciprocal_pairs is a METRIC and never gates. "A taught B and B taught A"
//     is historically impossible, so every pair is manufactured upstream by
//     identity collapse in `resolve` (da#356) — a thermometer for over-merge,
//     not a defect the loader can fix. Re-measure after da#356 lands. Do not
//     delete cycles (da#248).
//
// Cost: ~109s against the full stg graph, inside the 300s per-query validation
// budget (da#259). An overrun downgrades to a non-fatal WARN, never a hang.
// Single statement: no top-level `;` (see src/graph/validate.py::_split_statements).
CALL {
  MATCH (n:Narrator)-[:TRANSMITTED_TO]->(n)
  RETURN count(DISTINCT n.id) AS self_loops
}
CALL {
  MATCH (a:Narrator)-[:TRANSMITTED_TO]->(b:Narrator)
  WHERE a.id < b.id AND EXISTS { MATCH (b)-[:TRANSMITTED_TO]->(a) }
  RETURN count(DISTINCT [a.id, b.id]) AS reciprocal_pairs
}
RETURN self_loops, reciprocal_pairs
