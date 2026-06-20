---
name: project_profile_source_bio_promote
description: data-acquisition — a narrators-only (rijal-DB) source loads ZERO Narrator nodes through the mention-driven disambiguator; needs bio-direct promotion to canonical
metadata: 
  node_type: memory
  type: project
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

In noorinalabs-data-acquisition, the resolve path that mints canonical Narrator
records (`src/resolve/disambiguate.py::run`) is **mention-driven**: it returns
`[]` early when `total_mentions == 0`, only emitting a canonical narrator when an
isnad *mention* resolves to a bio candidate. So a **profile-only source** (a
rijal database that contributes `narrators_bio_*.parquet` but no isnad chains/
mentions — e.g. Itqan da#92a) produces **zero `Narrator` nodes** if you only run
the existing pipeline.

Fix shipped (da#92a, PR da#110): `src/resolve/bio_promote.py::promote_bios_to_canonical`
promotes each bio directly to a canonical narrator keyed by the SAME identity the
disambiguator uses — `src/parse/identity.make_canonical_id` =
`nar:<uuid5(CANONICAL_NAMESPACE, normalized-name)>` (centralized in identity.py as
of da#92a; disambiguate's `_make_canonical_id` now delegates). It is merge-safe
(unions an existing canonical parquet, doesn't clobber), so it composes with a
later mention-driven run.

**Two load-bearing gotchas for any future narrator source (#93/#94, other rijal):**
1. The graph loader `graph/load_nodes._load_narrators` reads
   `narrators_canonical.parquet` from the **staging** dir, but `disambiguate.run`
   writes it to the **curated/output** dir. Point the promoter at *staging* or the
   load finds nothing. (Pre-existing staging-vs-curated wiring quirk — flagged for
   follow-up.)
2. Itqan licensing: upstream repo (github R3GENESI5/Itqan) ships NO LICENSE →
   all-rights-reserved by default. Owner CLEARED it (2026-06-12) for non-profit
   use on the basis that we ingest FACTS re-expressed in our schema (not their
   files) + full provenance makes it removable. See [[project_data_pipeline_architecture]].

Node-level provenance (da#92a, PR#110): `Narrator` nodes now carry a `source_ids`
list (`<corpus>:<bare-id>`) via `_NARRATOR_MERGE` — so a corpus is removable on the
graph: `MATCH (n:Narrator) WHERE any(s IN n.source_ids WHERE s STARTS WITH 'itqan:')
DETACH DELETE n`. Before this, nodes had only a bare `external_id` and NO corpus tag,
so the owner's `{source_corpus:'itqan'}` delete would have matched zero — provenance
lived only in the staging parquet. Use the `source_ids` list (not a scalar
`source_corpus`) because a canonical narrator can be multi-source after dedup.
