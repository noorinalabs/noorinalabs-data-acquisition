// da#248/da#250 — over-merge offenders, ranked by reciprocal-partner count.
//
// DIAGNOSTIC, not a validation gate. This file lives under queries/analysis/ (NOT
// queries/validation/) on purpose: src.graph.validate.run_validation globs only
// queries/validation/*.cypher. A ranking query is *expected* to return rows.
//
// READ-ONLY: a single MATCH/WITH/RETURN — no writes, deletes, or fabrication.
//
// TERMINATION — the point of the rewrite:
//   The previous form was `MATCH path = (n:Narrator)-[:TRANSMITTED_TO*1..20]->(n)`
//   with no LIMIT, and it does not terminate on this graph. Not at *1..20, and not
//   even at *1..2, which was killed at 150s having emitted no rows at all. It was
//   therefore unrunnable, and the "uncapped cycle total" its header promised was
//   never obtainable from it. Replaced below by a degree-bounded formulation over
//   reciprocal (length-2) cycles, which completes in ~110s at full stg scale.
//
// WHAT IT ANSWERS:
//   Which canonical Narrator nodes sit in the most reciprocal pairs — i.e. which
//   identities were collapsed hardest. "A taught B and B taught A" is historically
//   impossible, so a high `reciprocal_partners` marks one node standing in for
//   several distinct men. Bare mononyms dominate: measured on stg 2026-07-09 the
//   head was عبد الله (829), سفيان (422), محمد (419), شعبة (360). The single سفيان
//   node merges Sufyān al-Thawrī (d. 161 AH, Kufa) with Sufyān ibn ʿUyayna
//   (d. 198 AH, Mecca) — two men a generation apart. See src/resolve/mononym_split.py
//   and da#356.
//
// The LIMIT below is a RANKING HEAD, not a population count: it caps how many
// offenders are displayed, and no total is derived from it. For the population use
// queries/validation/chain_integrity.cypher, which counts exactly and without a
// LIMIT (23,139 reciprocal pairs over 5,803 narrators as of 2026-07-09).
MATCH (n:Narrator)-[:TRANSMITTED_TO]->(m:Narrator)
WHERE n.id <> m.id AND EXISTS { MATCH (m)-[:TRANSMITTED_TO]->(n) }
WITH n, count(DISTINCT m.id) AS reciprocal_partners
RETURN n.id                 AS canonical_id,
       n.name_ar            AS name_ar,
       n.name_ar_normalized AS name_ar_normalized,
       n.generation         AS generation,
       n.mention_count      AS mention_count,
       reciprocal_partners
ORDER BY reciprocal_partners DESC, canonical_id ASC
LIMIT 200
