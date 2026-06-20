---
name: project_staging_graph_load_transport
description: "How to load data-acquisition parquet into staging/prod Neo4j — backend-network-only, bake-deps-then-attach loader image."
metadata: 
  node_type: memory
  type: project
  originSessionId: d8acc7c0-91ac-412b-b312-da38817b1614
---

Staging (and prod) Neo4j (`bolt://neo4j:7687`, container `noorinalabs-neo4j-1`, neo4j:5-community) is reachable ONLY on the docker network `noorinalabs_backend`, which has **no host-published bolt port** and **no internet egress**. Consequence: a loader container attached to that network cannot `pip install` at runtime, and you cannot SSH-tunnel a host port to it.

Canonical mechanism (da#174, PR#180): a self-contained loader image with deps+code baked at BUILD time (default bridge has egress), then RUN attached to `noorinalabs_backend`. Committed in `noorinalabs-data-acquisition`: `Dockerfile.load` + `scripts/load_staging.sh`. The script rsyncs node-bearing parquet to `/tmp/da174-load` on staging, builds the image, runs the loader on the backend net, and reads the neo4j password from the container's `NEO4J_AUTH` env (`docker exec noorinalabs-neo4j-1 printenv NEO4J_AUTH | cut -d/ -f2`) — no secret committed. Image left as `noorinalabs-graph-load:da174` for reuse.

- The deployed isnad-graph API image canNOT be reused as the loader: it ships the neo4j driver but NOT pyarrow (the Parquet loaders need it).
- `isnad-ingest load --nodes-only` = Hadith/Collection/Grading/Chain nodes, no edges, no dependency on the shared `narrators_canonical.parquet` (resolve output) → per-source, idempotent (MERGE on corpus-namespaced ids), conflict-free for parallel per-source loads. `LOAD_ARGS="load"` does the full nodes+edges load (needs the shared resolve first).
- ssh as `deploy@noorinalabs-stg`; `/home/deploy` is root-owned (NOT writable) — use `/tmp` (on the 190G `/`, not tmpfs). Staging load is owner-sanctioned (da#73 pattern); PROD apply (deploy#470) is owner-sign-off only.
- The ONE combined narrator-resolve + chain/transmission/PARALLEL_OF edge load over the union of all sources is the shared critical-path step feeding prod cutover; run it once as a no-clobber merge after all source parquet exist. See [[project_narrators_two_producers]], [[project_hadith_id_double_prefix]], [[project_staging_unreachable_from_sandbox]].
