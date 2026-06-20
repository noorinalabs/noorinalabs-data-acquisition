---
name: feedback_generate_source_id_failfast
description: "da#82 generate_source_id() fail-fast crashes any key built with a non-SourceCorpus namespace (e.g. bio_id \"kaggle_narrators\"); latent until that path runs"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 090bf6d5-0d19-47c9-9b85-67bfff1c5396
---

After da#82 (ig/da PR #102), `generate_source_id(corpus, ...)` raises `ValueError`
when `corpus` is not in `SourceCorpus`. Several adapters (mis)used it to mint
**non-hadith provenance keys** — e.g. sanadset's narrator-bio `bio_id` used
`generate_source_id(_BIO_SOURCE="kaggle_narrators", ...)`, and `kaggle_narrators`
is NOT a corpus. So the new fail-fast aborts the WHOLE parse the moment that path
runs (for sanadset: the moment a `narrators/` dir is present — i.e. the real
acquisition path).

**Why:** It is latent because the unit/parse tests don't exercise the branch
(sanadset parse tests create no `narrators/` dir), so it passes CI green while
being broken on real data. Fixed in da#89 (PR #106) by building the bio key
directly with `ID_DELIMITER.join([...])` — provenance/aux keys are a different
namespace from corpus-gated hadith/collection `source_id`s and must bypass
`generate_source_id`.

**How to apply:** When reviewing/authoring any per-source light-up post-da#82
(#83, #85 thaqalayn, #89, future ones), grep the adapter+parser for
`generate_source_id(` and confirm the first arg is a real `SourceCorpus` value;
any bio/narrator/aux key namespaced by a *source label* (kaggle_narrators,
kaggle, etc.) must use a direct delimiter join, not the corpus-gated builder.
Pair the fix with a test that actually creates the bio/aux input so the branch
runs. Related: [[feedback_appears_in_merge_null]],
[[project_hadith_id_double_prefix]].
