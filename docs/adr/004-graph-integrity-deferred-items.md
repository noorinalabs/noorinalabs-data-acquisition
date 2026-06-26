# ADR-004: Graph-integrity deferred-item contract decisions (da#153)

## Status: Accepted (contract decisions) — owner decisions itemized below (2026-06-25, P7W20, da#153)

> Follow-up to the da#148 integrity sweep (PR #150, which shipped the clean
> producer fixes: the `TRANSMITTED_TO` self-loop guard + `grade_normalized`). The
> five items below were **deliberately deferred** from PR #150 because each needs a
> data decision or upstream producer work — none is a one-line load-layer fix, and
> **none may be resolved by fabricating values at load time**. This ADR makes each
> contract decision explicit so nothing is silently dropped or silently invented.
> No remediation of live graph state executes on the basis of this ADR.

## Context

Production validation of the live graph (da#148 sweep, P5W1) surfaced five
graph-integrity findings that the producer/load layer cannot honestly resolve by
itself. The standing temptation with each is to **fabricate** a plausible value at
load time (a synthesized in-book ordinal, a placeholder narrator, a guessed
Arabic name). That is exactly the wrong move for a scholarly corpus: a fabricated
ordinal or narrator is indistinguishable downstream from a sourced one, and
silently corrupts provenance. The governing principle (org memory "prefer correct
over expedient") is to encode an **explicit-null / explicit-contract** handling,
test it, size the gap with a read-only inventory, and escalate the genuine data
decisions to the owner.

## Decisions — per item

### Item #1 — 86 null `hadith_number_in_book` (in-book ordinal)

**Contract decision (made here, no owner input required): EXPLICIT NULL where the
in-book ordinal is genuinely unknown.** The staging `hadith_number` column is the
in-book ordinal that flows to `APPEARS_IN.hadith_number_in_book` (da#77). Scraped
sources that carry only the collection-wide reference number leave it null on
purpose (project memory "hadith number: collection-ref vs in-book ordinal"). The
load layer already honours this correctly: `_APPEARS_IN_QUERY`
(`src/graph/load_edges.py`) SETs the property with the coalesce-preserve form
`r.hadith_number_in_book = coalesce(row.hadith_number, r.hadith_number_in_book)`
*after* a property-less MERGE, so a null ordinal leaves the edge property **absent**
rather than fabricated (and a re-load never clears a value already present).

- **Encoded + pinned by:** `hadith_number_coverage` informational check in
  `src/parse/validate.py::_hadith_checks` (always PASS — a null in-book ordinal is
  a tracked, contractual state, not a failure), with
  `tests/test_parse/test_validate.py::...::test_in_book_ordinal_explicit_null_is_contractual`.
  The pre-existing edge-level pins
  (`test_appears_in_edge_uses_canonical_property_key`,
  `test_appears_in_merge_has_no_property_key_null_unsafe`) guard the load contract.
- **Owner decision still open (per-source):** whether any specific source can have
  its in-book ordinal *derived* (e.g. from page order or a secondary field) rather
  than left null. That is a per-scraper data decision in the parse lane
  (da#72/Alejandra's lane), **not** a load-layer change. Sized by block #1 of the
  inventory query.

### Item #2 — 51 Hadith with no `NARRATED` edge

**Contract decision (made here): no fabrication.** A hadith whose mentions carry
no `canonical_narrator_id` has no resolved position-0 narrator, so `_load_narrated`
(`src/graph/load_edges.py`) skips it — it does **not** invent a synthetic narrator
to attribute narration to. This is an upstream NER/mention-coverage gap (related to
the da#146 segmentation work; re-measure after da#151 lands), addressable only by
producer-side mention extraction.

- **Encoded + pinned by:**
  `tests/test_graph/test_load_edges.py::TestLoadNarrated::test_unresolved_chain_emits_no_narrated_no_fabrication`
  — an unresolved chain yields zero `NARRATED` edges and issues no write.
- **Owner / producer work still open:** improving NER/mention coverage so these
  hadiths gain a resolved chain. Sized by block #2 of the inventory query.

### Item #3 — 8 orphan `muhaddithat` bio-only narrators

**Owner decision required — link vs drop.** These are narrators promoted from a bio
source (`muhaddithat`) without mention-linking, so they have no graph edges. The
two options — (a) mention-link them onto the hadith they transmit, or (b) drop them
as bio-only noise — are a data-curation call the load layer must not make
unilaterally (dropping loses sourced biographical figures; auto-linking risks a
wrong attribution). The existing `queries/validation/orphan_narrators.cypher`
already detects them; block #3/#3b of the new inventory query enumerates them with
a bounded sample for the decision.

- **Status:** escalated to the owner in the PR body. No fabrication; no drop; no
  auto-link until decided.

### Item #4 — `grade_normalized` parity in the streaming path

**Cross-repo follow-up, not a data-acquisition change.** PR #150 added
`normalize_grade()` + the `grade_normalized` property on the **batch** load path
(`src/graph/load_nodes.py`). The Kafka streaming worker lives in a different repo
(`noorinalabs-isnad-ingest-platform`, `workers/ingest`) and must mirror the same
normalization so batch and stream converge. That is a parity change in the other
repo; this repo's batch path is already correct.

- **Status:** flagged for a cross-repo parity follow-up in the PR body. Nothing to
  change in `noorinalabs-data-acquisition`.

### Item #5 — Collection metadata gap (blank Arabic name / `riyadussalihin`)

**Data-enrichment decision, source-by-source.** Some `Collection` nodes carry a
blank/null Arabic name or lack the expected-count enrichment (`riyadussalihin` is
the named instance from the sweep). The correct value is sourced metadata, not a
guess, so the load layer leaves it null rather than fabricating an Arabic name.
Filling it is a per-collection enrichment in the relevant parser's `collections_*`
emission.

- **Status:** owner/enrichment decision; sized by block #5 of the inventory query
  (generalized to "any Collection missing `name_ar` or `expected_count`"). No
  fabrication.

## Inventory (read-only, no remediation)

`queries/validation/graph_integrity_deferred_inventory.cypher` is a read-only,
multi-block query (mirrors `sanadset_orphan_inventory.cypher` from ADR-003) that
sizes items #1, #2, #3, and #5 per corpus. Running it never writes, deletes, or
fabricates — the numbers it returns are a tracked, owner-accepted state, not a
remediation trigger.

## Consequences

**Positive.** Every deferred item now has an explicit, documented contract; the
no-fabrication stance is pinned by tests at both the staging (`hadith_number`
coverage) and load (`NARRATED` no-fabrication, `APPEARS_IN` explicit-null) layers,
so a future change cannot silently turn an explicit-null into a fabricated value or
a hard failure that pressures fabrication. The owner decisions (#3 link-vs-drop,
#1 per-source derivation, #5 enrichment) and the cross-repo parity (#4) are
surfaced rather than buried.

**Negative / trade-offs.** The three owner-decision items remain open data
decisions — the graph keeps the null/orphan state until the owner picks a
direction. That is the correct conservative posture for a provenance-sensitive
corpus, but it means the counts (86 / 51 / 8 / metadata gaps) persist as a known,
tracked state until those decisions land.

## Related

- da#153 — this issue: graph-integrity sweep deferred-item contract decisions.
- da#148 / PR #150 — the producer fixes that shipped (self-loop guard + grade
  normalization); these five items were split out as the deferred follow-up.
- da#77 — in-book ordinal vs collection-wide reference; null-safe APPEARS_IN.
- da#146 / da#151 — narrator segmentation / mention coverage (item #2 upstream).
- ADR-003 — `sanadset` orphan remediation; sibling read-only inventory pattern.
- `noorinalabs-isnad-graph-ingestion` streaming worker — item #4 grade parity.
- `queries/validation/graph_integrity_deferred_inventory.cypher` — read-only sizing.
