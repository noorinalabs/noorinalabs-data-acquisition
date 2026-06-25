// sanadset orphan-node / narrator-pollution inventory (da#202, P7W19).
//
// READ-ONLY. Every statement is MATCH/RETURN/count — NO writes, NO deletes.
// Run each block independently (split on the blank lines) to characterize the
// sanadset blast radius BEFORE any owner-run per-source purge. These numbers
// quantify Path A loss and Path B scope for ADR-003.

// 1. Hadith nodes per source_corpus, and how many are collection-linked.
//    Confirms the orphan scale: sanadset hadiths with 0 APPEARS_IN edges.
MATCH (h:Hadith)
OPTIONAL MATCH (h)-[a:APPEARS_IN]->(:Collection)
WITH h.source_corpus AS source,
     count(DISTINCT h) AS hadiths,
     count(a) AS appears_in_edges
RETURN source, hadiths, appears_in_edges,
       hadiths - appears_in_edges AS orphan_hadiths
ORDER BY hadiths DESC;

// 2. Collection nodes per source_corpus.
//    Expectation: sanadset has 0 Collection nodes (no collections_sanadset
//    parquet is ever emitted by parse_sanadset).
MATCH (c:Collection)
RETURN c.source_corpus AS source, count(c) AS collections
ORDER BY collections DESC;

// 3. Narrators reachable ONLY through sanadset hadiths (Path A would remove
//    these; they are the bulk of the polluted narrator table). A narrator is
//    sanadset-exclusive if every hadith it NARRATED has source_corpus sanadset.
MATCH (n:Narrator)-[:NARRATED]->(h:Hadith)
WITH n,
     count(h) AS total_h,
     sum(CASE WHEN h.source_corpus = 'sanadset' THEN 1 ELSE 0 END) AS sanadset_h
WHERE total_h = sanadset_h
RETURN count(n) AS sanadset_exclusive_narrators;

// 4. TRANSMITTED_TO edges whose endpoints are sanadset-exclusive narrators
//    (Path A collateral on the teacher-student graph). Bounded sample.
MATCH (a:Narrator)-[t:TRANSMITTED_TO]->(b:Narrator)
WHERE NOT EXISTS {
        MATCH (a)-[:NARRATED]->(h:Hadith) WHERE h.source_corpus <> 'sanadset'
      }
RETURN count(t) AS transmitted_to_from_sanadset_only_narrators;

// 5. Distinct sanadset collection_name values carried on Hadith nodes but with
//    no backing Collection node. These are the would-be collections Path B
//    must materialize from books.csv (book_id -> book -> collection slug).
MATCH (h:Hadith {source_corpus: 'sanadset'})
RETURN DISTINCT h.collection_name AS unlinked_collection_name,
       count(*) AS hadiths
ORDER BY hadiths DESC
LIMIT 100;
