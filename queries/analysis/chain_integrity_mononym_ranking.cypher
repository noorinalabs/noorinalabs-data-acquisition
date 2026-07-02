// da#248 — TRANSMITTED_TO cycle offenders, ranked by anchor canonical node.
//
// DIAGNOSTIC, not a validation gate. This file lives under queries/analysis/ (NOT
// queries/validation/) on purpose: src.graph.validate.run_validation globs only
// queries/validation/*.cypher and classifies any non-empty result as a FAILING
// check. A ranking query is *expected* to return rows, so it must stay out of that
// directory — it never degrades the shipped chain_integrity.cypher guard, it only
// quantifies what that guard caps at LIMIT 100.
//
// READ-ONLY: a single MATCH/WITH/RETURN — no writes, deletes, or fabrication.
//
// What it answers (da#248 acceptance item 1):
//   * Uncapped cycle total — sum(cycle_count) over all returned rows is the TRUE
//     cycle count that chain_integrity.cypher truncates at 100.
//   * Worst offenders — cycles grouped by the canonical Narrator node that anchors
//     them (the n where n = m closes the loop), ranked descending. A common,
//     under-disambiguated mononym (e.g. سفيان "Sufyan", which merges Sufyān
//     al-Thawrī and Sufyān ibn ʿUyayna into one node) surfaces at the top because
//     its single over-merged node back-edges into itself across generations.
//
// Cost note: bounded variable-length cycle detection (*1..20) over the full
// narrator graph is expensive. Run it as an offline diagnostic against a loaded
// graph (staging/prod reload), not on a hot path. The *1..20 bound mirrors the
// shipped chain_integrity.cypher so the two count the same population.
MATCH path = (n:Narrator)-[:TRANSMITTED_TO*1..20]->(n)
WITH n, count(path) AS cycle_count
RETURN n.id                 AS canonical_id,
       n.name_ar            AS name_ar,
       n.name_ar_normalized AS name_ar_normalized,
       n.generation         AS generation,
       cycle_count
ORDER BY cycle_count DESC, canonical_id ASC
