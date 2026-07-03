# DuckDB Parquet exploration (`make duck`)

A **dev-only** instrument for interactive SQL over the pipeline's staging and
curated Parquet files. It answers questions like "top blocks by member count" or
"mention_count distribution per canonical narrator" with one query instead of a
bespoke PyArrow script.

- Implementation: `src/tools/duck.py`
- Dependency: `duckdb` (uv **dev** group only — it never enters the runtime image)
- **Read-only** over `data/`: the helper only issues `CREATE VIEW ... read_parquet`,
  never a write. Safe to run against the directories a live `resolve` owns.

## Running

```bash
make duck                                  # interactive SQL REPL
make duck QUERY="SELECT count(*) FROM curated_narrators_canonical"   # one-shot

# equivalently, directly:
uv run python -m src.tools.duck            # REPL
uv run python -m src.tools.duck -c "SELECT ..."          # one-shot table
uv run python -m src.tools.duck -c "SELECT ..." --csv    # one-shot CSV to stdout
uv run python -m src.tools.duck --list                   # list views, then exit
uv run python -m src.tools.duck --staging DIR --curated DIR   # point at other dirs
```

In the REPL, `.views` lists the registered views and `.quit` exits.

## Views

Data-dir paths come from `src/config.py` settings (`DATA_STAGING_DIR`,
`DATA_CURATED_DIR`), never hardcoded. View names are derived from file stems:

- **Per-file** view for every `*.parquet`: `<layer>_<stem>` — e.g.
  `curated_narrators_canonical`, `curated_narrator_mentions_resolved`,
  `staging_hadiths_sanadset`, `staging_narrators_bio_itqan`.
- **Combined** view for a multi-file dataset — ≥2 files that share a leading name
  prefix *and an identical schema* (true shards of one producer): `<layer>_<prefix>`,
  e.g. `staging_hadiths` (all `hadiths_*` shards unioned), `staging_collections`,
  `staging_narrators_bio`, `staging_network_edges`, `staging_narrator_mentions`.

The schema-identity check is what keeps distinct datasets that merely share a name
prefix apart: `narrator_aliases_*` and `narrator_mentions_*` both begin `narrator`
but have different schemas, so no bogus `staging_narrator` view is created.

## Example queries

```sql
-- 1. Top-20 canonical narrators by mention_count
SELECT canonical_id, name_ar, mention_count
FROM curated_narrators_canonical
ORDER BY mention_count DESC
LIMIT 20;
```

```sql
-- 2. mention_count distribution (how many narrators are singletons vs recurring)
SELECT
  CASE WHEN mention_count = 1 THEN '1'
       WHEN mention_count BETWEEN 2 AND 4 THEN '2-4'
       WHEN mention_count BETWEEN 5 AND 19 THEN '5-19'
       ELSE '20+' END AS bucket,
  count(*) AS narrators
FROM curated_narrators_canonical
GROUP BY bucket
ORDER BY min(mention_count);
```

```sql
-- 3. Name-length histogram over canonical names (token count)
SELECT len(string_split(name_ar_normalized, ' ')) AS tokens, count(*) AS narrators
FROM curated_narrators_canonical
WHERE name_ar_normalized IS NOT NULL
GROUP BY tokens
ORDER BY tokens;
```

```sql
-- 4. Blocking-token block sizes: how many canonical narrators carry each name
--    token (an approximation of the fuzzy_cluster blocking pool). See da#271 for
--    the connector-token exclusion and pair-contribution accounting.
WITH tok AS (
  SELECT canonical_id, unnest(string_split(name_ar_normalized, ' ')) AS token
  FROM curated_narrators_canonical
  WHERE name_ar_normalized IS NOT NULL
)
SELECT token, count(*) AS block_members
FROM tok
WHERE length(token) >= 2
GROUP BY token
ORDER BY block_members DESC
LIMIT 50;
```
