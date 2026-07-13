// Find narrator nodes with no relationships, EXCLUDING biography-only narrators.
// Expected: 0 orphans in a healthy graph
// A `biographical_only` narrator (da#370) is a real person carried by a rijal /
// biographical source with no surviving isnad mention. Such narrators are kept on
// purpose ("tag & keep") and are zero-degree BY DESIGN, so they are not orphans —
// excluding them keeps this contract meaningful for a genuine orphan regression.
// coalesce guards a legacy node loaded before the attestation property existed.
MATCH (n:Narrator)
WHERE NOT (n)--()
  AND coalesce(n.attestation, '') <> 'biographical_only'
RETURN n.id AS narrator_id, n.name_en AS name
ORDER BY n.id
