---
name: project_itqan_license_proceed
description: "Owner approved using the Itqan narrator source (no upstream license / all-rights-reserved) — proceed; it's for a non-profit and is cleanly removable via source_corpus=itqan provenance if ever problematic."
metadata: 
  node_type: memory
  type: project
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

Owner decision (2026-06-12, P4W3, da#92a / PR #110): **proceed** with the Itqan narrator source despite it shipping **no upstream license** (= all-rights-reserved by default).

**Why it's acceptable:**
- The project is a **non-profit** — owner is comfortable with the residual risk.
- We extract **facts** (classical/public-domain hadith narrator data — names, dates, jarh-wa-ta'dil gradings) and **re-express them in our own canonical schema** (our `nar:` ids, our node structure), rather than redistributing Itqan's files/compilation verbatim. Copyright protects creative expression/compilation, not facts.
- **Cleanly removable:** Ivana made this true on the GRAPH (da#92a/#110 commit cd38ece) — Narrator nodes carry a `source_ids` **list** (a canonical narrator can be multi-source after dedup). Correct removal query: `MATCH (n:Narrator) WHERE any(s IN n.source_ids WHERE s STARTS WITH 'itqan:') DETACH DELETE n`. (NOT the scalar `{source_corpus:'itqan'}` form — that matched ZERO Narrator nodes; provenance was only in staging parquet before cd38ece.) Owner: "I'd have no problem removing it later."

**How to apply:** don't re-litigate the Itqan license as a blocker on future PRs. Keep the provenance tagging (`source_corpus` + `external_id`) intact on any source so the same remove-if-problematic posture holds for other no/unclear-license sources. The general posture: facts-in-our-schema + per-source provenance tagging = acceptable + reversible. Itqan = largest narrator source (115,735 profiles; 12,820 in the first scale run).
