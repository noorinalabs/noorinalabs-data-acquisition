// Regression guard for da#325: every TRANSMITTED_TO.hadith_id must reference an
// existing Hadith node. The chain endpoint GET /graph/hadith/{id}/chain joins
// TRANSMITTED_TO.hadith_id against Hadith.id; a raw (non-canonical) edge id
// (sanadset:sanadset:...) matches no Hadith.id (hdt:sanadset:...), so the join
// returns empty for every hadith. Report distinct edge ids with no Hadith node.
// Expected: 0 rows in a healthy graph.
MATCH ()-[t:TRANSMITTED_TO]->()
WHERE t.hadith_id IS NOT NULL AND t.hadith_id <> ""
WITH DISTINCT t.hadith_id AS hadith_id
OPTIONAL MATCH (h:Hadith {id: hadith_id})
WITH hadith_id, h
WHERE h IS NULL
RETURN hadith_id
ORDER BY hadith_id
LIMIT 100
