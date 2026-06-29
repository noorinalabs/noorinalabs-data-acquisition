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
LOAD_CONTAINER="${LOAD_CONTAINER:-da174-graph-load}"
# Default: nodes only (Hadith/Collection/Grading/Chain) — no edges, no dependency
# on the shared narrator-resolve output. Set LOAD_ARGS="load" for a full load.
LOAD_ARGS="${LOAD_ARGS:-load --nodes-only}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[1/5] sync code + node-bearing staging parquet -> ${SSH_HOST}:${REMOTE_DIR}"
ssh "$SSH_HOST" "mkdir -p ${REMOTE_DIR}/data/staging ${REMOTE_DIR}/data/curated"
rsync -az pyproject.toml uv.lock Dockerfile.load "${SSH_HOST}:${REMOTE_DIR}/"
rsync -az --delete src queries "${SSH_HOST}:${REMOTE_DIR}/"
# The curated resolve outputs (narrators_canonical + resolved mentions) are
# needed by BOTH load modes, not just the full load:
#   * Narrator nodes load from curated/narrators_canonical.parquet.
#   * Chain nodes' narrator_ids come from curated/narrator_mentions_resolved*.parquet
#     (the canonical_narrator_id column lives there, not in the raw staging
#     mentions). Reading staging produced hollow chains — the #723 "chains empty"
#     defect (load_nodes._load_chains).
# So ship curated UNCONDITIONALLY. The edge-bearing *staging* parquet
# (parallel_links, network_edges, staging mentions) is what's full-load-only.
rsync -az \
    data/curated/narrators_canonical.parquet \
    data/curated/narrator_mentions_resolved.parquet \
    data/curated/narrator_mentions_resolved_muhaddithat.parquet \
    "${SSH_HOST}:${REMOTE_DIR}/data/curated/"
if printf '%s' "$LOAD_ARGS" | grep -q -- '--nodes-only'; then
    STAGING_INCLUDES=(--include='hadiths_*.parquet' --include='collections_*.parquet' --exclude='*')
else
    STAGING_INCLUDES=(
        --include='hadiths_*.parquet' --include='collections_*.parquet'
        --include='narrator_mentions_*.parquet' --include='network_edges_*.parquet'
        --include='parallel_links.parquet' --exclude='*'
    )
fi
rsync -az --delete "${STAGING_INCLUDES[@]}" \
    data/staging/ "${SSH_HOST}:${REMOTE_DIR}/data/staging/"

echo "[2/5] build loader image on ${SSH_HOST}"
ssh "$SSH_HOST" "cd ${REMOTE_DIR} && docker build -f Dockerfile.load -t ${IMAGE} ."

echo "[3/5] run loader (DETACHED) on the ${BACKEND_NET} network (${LOAD_ARGS})"
# Run the loader detached (docker run -d) so its lifetime is owned by dockerd,
# NOT this SSH session. A multi-hour FOREGROUND `docker run` over SSH is killed
# by any broken pipe / idle-timeout: #723's ~2h stage edge-load died mid-
# PARALLEL_OF to `client_loop: send disconnect: Broken pipe` (rc=255), taking
# the --rm container with it. Detached + short-poll (step 4) survives SSH drops.
# Idempotent MERGE loaders mean a re-run after such a drop simply converges.
# shellcheck disable=SC2087  # heredoc is intentionally expanded on the remote.
ssh "$SSH_HOST" bash -s <<REMOTE
  set -euo pipefail
  PW=\$(docker exec ${NEO4J_CONTAINER} printenv NEO4J_AUTH | cut -d/ -f2)
  docker rm -f ${LOAD_CONTAINER} >/dev/null 2>&1 || true
  docker run -d --name ${LOAD_CONTAINER} --network ${BACKEND_NET} \
    -e NEO4J_URI=bolt://neo4j:7687 -e NEO4J_USER=neo4j -e NEO4J_PASSWORD="\$PW" \
    -v ${REMOTE_DIR}/data:/app/data \
    ${IMAGE} ${LOAD_ARGS}
REMOTE

echo "[3b/5] poll loader to completion (short SSH calls — no held session)"
while :; do
  status="$(ssh "$SSH_HOST" "docker inspect -f '{{.State.Status}}' ${LOAD_CONTAINER} 2>/dev/null" || echo ssh-blip)"
  case "$status" in
    exited)
      rc="$(ssh "$SSH_HOST" "docker inspect -f '{{.State.ExitCode}}' ${LOAD_CONTAINER}" 2>/dev/null || echo '?')"
      echo "  loader container exited rc=${rc} at $(date -u +%H:%M:%SZ)"
      ssh "$SSH_HOST" "docker logs --tail 30 ${LOAD_CONTAINER}" 2>&1 | sed 's/^/    /' || true
      [ "$rc" = "0" ] || { echo "LOAD FAILED (rc=${rc}) — graph is partial; re-run converges (MERGE)."; exit 1; }
      break ;;
    running)  echo "  ... loading ($(date -u +%H:%M:%SZ))"; sleep 60 ;;
    ssh-blip) echo "  ... ssh blip, container unaffected; retry ($(date -u +%H:%M:%SZ))"; sleep 30 ;;
    *)        echo "  ... status=${status} ($(date -u +%H:%M:%SZ))"; sleep 30 ;;
  esac
done

echo "[4/5] read back source_corpus distribution"
ssh "$SSH_HOST" bash -s <<REMOTE
  set -euo pipefail
  PW=\$(docker exec ${NEO4J_CONTAINER} printenv NEO4J_AUTH | cut -d/ -f2)
  docker exec ${NEO4J_CONTAINER} cypher-shell -u neo4j -p "\$PW" \
    'MATCH (h:Hadith) RETURN h.source_corpus AS source_corpus, count(*) AS hadiths ORDER BY hadiths DESC'
REMOTE

echo "[5/5] done."
