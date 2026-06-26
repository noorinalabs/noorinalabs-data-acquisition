// Graph-integrity deferred-item inventory (da#153, P7W20). Companion to
// docs/adr/004-graph-integrity-deferred-items.md.
//
// READ-ONLY. Every statement is MATCH/OPTIONAL MATCH/RETURN/count — NO writes,
// NO deletes, NO fabrication. Run each block independently (split on the blank
// lines) to SIZE the five deferred items before any per-source data decision.
// These numbers are a tracked, owner-accepted state — they are NOT remediated by
// running this file.

// 1. In-book ordinal (item #1). APPEARS_IN edges whose hadith_number_in_book is
//    absent (the explicit-null contract, da#77/ADR-004) vs present, per corpus.
//    The "missing" count is the population a future per-source ordinal-derivation
//    decision must cover; it is contractual, not a defect.
MATCH (h:Hadith)-[r:APPEARS_IN]->(:Collection)
WITH h.source_corpus AS source,
     count(r) AS appears_in_edges,
     sum(CASE WHEN r.hadith_number_in_book IS NULL THEN 1 ELSE 0 END) AS null_in_book_ordinal
RETURN source, appears_in_edges, null_in_book_ordinal,
       appears_in_edges - null_in_book_ordinal AS with_in_book_ordinal
ORDER BY null_in_book_ordinal DESC;

// 2. NARRATED coverage gap (item #2). Hadith nodes with no inbound NARRATED edge
//    — the upstream NER/mention-coverage gap. No position-0 narrator was resolved,
//    so no edge is fabricated. Sized per corpus.
MATCH (h:Hadith)
WHERE NOT (:Narrator)-[:NARRATED]->(h)
RETURN h.source_corpus AS source, count(h) AS hadiths_without_narrated
ORDER BY hadiths_without_narrated DESC;

// 3. Orphan bio-only narrators (item #3). Narrator nodes with NO relationships at
//    all (no NARRATED, no TRANSMITTED_TO, no STUDIED_UNDER) — promoted from a bio
//    source without mention-linking. The muhaddithat orphans (Abu Bakr, Fatimah,
//    ...) are the owner-decision population: link (mention-linking) or drop.
MATCH (n:Narrator)
WHERE NOT (n)--()
RETURN coalesce(n.source_corpus, 'unknown') AS source,
       count(n) AS orphan_narrators
ORDER BY orphan_narrators DESC;

// 3b. Bounded sample of the orphan narrators for the owner link-vs-drop decision.
MATCH (n:Narrator)
WHERE NOT (n)--()
RETURN n.id AS narrator_id, n.name_en AS name_en, n.name_ar AS name_ar,
       n.source_corpus AS source
ORDER BY n.id
LIMIT 50;

// 5. Collection metadata gaps (item #5). Collection nodes missing an Arabic name
//    (blank or null) or missing the expected-count enrichment. riyadussalihin is
//    the named instance in the da#148 sweep; this generalizes the check.
MATCH (c:Collection)
WHERE c.name_ar IS NULL OR trim(c.name_ar) = '' OR c.expected_count IS NULL
RETURN c.id AS collection_id, c.name_en AS name_en, c.name_ar AS name_ar,
       c.expected_count AS expected_count, c.source_corpus AS source
ORDER BY c.id;
