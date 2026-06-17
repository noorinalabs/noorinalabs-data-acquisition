#!/usr/bin/env bash
# Build + run the one-off graph-load image against a remote cluster's Neo4j.
#
# da#174 — multi-source staging graph load. The cluster Neo4j (bolt://neo4j:7687)
# is reachable only on the internet-isolated `noorinalabs_backend` docker network,
# so we cannot tunnel a host port to it and cannot pip-install inside a container
# attached to that network. Instead we build a self-contained loader image
# (Dockerfile.load) where deps + code are baked at build time, ship the staging
# Parquet, and run the loader attached to `noorinalabs_backend`.
#
# Idempotent: the loaders MERGE on stable, corpus-namespaced ids, so re-running is
# safe. Reversible: every node carries `source_corpus`, so a source is cleanly
# removable with `MATCH (n {source_corpus:'<corpus>'}) DETACH DELETE n`.
#
# The Neo4j password is read from the running neo4j container's NEO4J_AUTH on the
# remote host, so no secret is committed or passed through this machine's shell
# history.
#
# Usage:
#   scripts/load_staging.sh                 # nodes-only load to staging
#   LOAD_ARGS="load" scripts/load_staging.sh   # full nodes+edges load
#   SSH_HOST=noorinalabs-prod scripts/load_staging.sh   # target prod (after owner go)
set -euo pipefail

SSH_HOST="${SSH_HOST:-noorinalabs-stg}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-noorinalabs-neo4j-1}"
BACKEND_NET="${BACKEND_NET:-noorinalabs_backend}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/da174-load}"
IMAGE="${IMAGE:-noorinalabs-graph-load:da174}"
# Default: nodes only (Hadith/Collection/Grading/Chain) — no edges, no dependency
# on the shared narrator-resolve output. Set LOAD_ARGS="load" for a full load.
LOAD_ARGS="${LOAD_ARGS:-load --nodes-only}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[1/5] sync code + node-bearing staging parquet -> ${SSH_HOST}:${REMOTE_DIR}"
ssh "$SSH_HOST" "mkdir -p ${REMOTE_DIR}/data/staging ${REMOTE_DIR}/data/curated"
rsync -az pyproject.toml uv.lock Dockerfile.load "${SSH_HOST}:${REMOTE_DIR}/"
rsync -az --delete src queries "${SSH_HOST}:${REMOTE_DIR}/"
# Ship only the node-bearing parquet (hadiths_/collections_) for a nodes-only load.
rsync -az --delete \
    --include='hadiths_*.parquet' --include='collections_*.parquet' --exclude='*' \
    data/staging/ "${SSH_HOST}:${REMOTE_DIR}/data/staging/"

echo "[2/5] build loader image on ${SSH_HOST}"
ssh "$SSH_HOST" "cd ${REMOTE_DIR} && docker build -f Dockerfile.load -t ${IMAGE} ."

echo "[3/5] run loader on the ${BACKEND_NET} network (${LOAD_ARGS})"
# shellcheck disable=SC2087  # heredoc is intentionally expanded on the remote.
ssh "$SSH_HOST" bash -s <<REMOTE
  set -euo pipefail
  PW=\$(docker exec ${NEO4J_CONTAINER} printenv NEO4J_AUTH | cut -d/ -f2)
  docker run --rm --network ${BACKEND_NET} \
    -e NEO4J_URI=bolt://neo4j:7687 -e NEO4J_USER=neo4j -e NEO4J_PASSWORD="\$PW" \
    -v ${REMOTE_DIR}/data:/app/data \
    ${IMAGE} ${LOAD_ARGS}
REMOTE

echo "[4/5] read back source_corpus distribution"
ssh "$SSH_HOST" bash -s <<REMOTE
  set -euo pipefail
  PW=\$(docker exec ${NEO4J_CONTAINER} printenv NEO4J_AUTH | cut -d/ -f2)
  docker exec ${NEO4J_CONTAINER} cypher-shell -u neo4j -p "\$PW" \
    'MATCH (h:Hadith) RETURN h.source_corpus AS source_corpus, count(*) AS hadiths ORDER BY hadiths DESC'
REMOTE

echo "[5/5] done."
