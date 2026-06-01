#!/usr/bin/env bash
# Apply / re-apply the noorinalabs-data-acquisition default-branch protection ruleset.
#
# Phase-3 end-state criterion #4 (noorinalabs-main#322): CI failures block all
# merges on every repo's default branch, enforced server-side by GitHub (not
# only by the Hook 4 comment-gate). This script is the OWNER/ADMIN-gated apply
# step: it POSTs (or, if a same-named ruleset already exists, PUT-updates) the
# canonical ruleset committed alongside it (ruleset-main.json).
#
# ALREADY APPLIED — data-acquisition is the W13 #322 PILOT. The ruleset is live
# on main (id 17091263, enforcement: active). ruleset-main.json in this directory
# DOCUMENTS that live ruleset; the committed PR artifact is the SPEC + this
# script + the reconciled payload. Running this script is NOT required for #322
# to be met for this repo — it is here for parity with the other repos and for a
# deliberate idempotent re-apply if the documented shape ever changes. Because a
# same-named ruleset already exists, this script PUT-updates it (no duplicate).
#
# Why ruleset apply is an owner/admin step (when it IS run):
#   Creating/updating a repository ruleset requires repo-admin permission, which
#   the agent gh principal (parametrization) does not hold for this purpose, and
#   applying default-branch protection while a wave-branch PR is in flight can
#   block our own merges. So the owner runs it from a window with no in-flight
#   default-branch merge (post-wave-wrapup is the safe window), then
#   read-back-verifies.
#
# Design notes (see SPEC.md in this directory + charter pull-requests.md
# § Org-Wide Branch Protection):
#   * required_approving_review_count: 0 — GitHub's "require approvals" counts
#     FORMAL PR reviews, which our team structurally cannot produce (the gh auth
#     principal IS the PR author, so a formal self-approval 422s). Reviewer-count
#     enforcement stays with Hook 4 (validate_pr_review). A "require 1 approval"
#     rule would deadlock every merge.
#   * Required status checks: data-acquisition has unconditional PR CI, so the
#     ruleset hard-requires its gate contexts (job NAMES, not workflow names):
#     `Lint`, `Type Check`, `Test`, `Integration Tests`. Confirm them against
#     live check-runs at apply time
#     (`gh api repos/<repo>/commits/<default-sha>/check-runs`) — CI job names can
#     change. The `Pre-commit ⇄ CI sync-drift gate` job added in W14 is an
#     additional gate intentionally NOT in the required set (mirror-drift is a
#     soft-direction process check); add its context to ruleset-main.json only if
#     the owner decides to hard-require it.
#   * Repository-admin always-bypass (actor_id 5) keeps the orchestrator's
#     --admin wave→main wrapup merges + the charter single-reviewer/doc-sweep/
#     emergency exceptions working. The hook-side ADMIN_MERGE_EXCEPTION gate
#     (validate_pr_ci_status) audits every such bypass.
#
# Usage:
#   ./apply-ruleset.sh                 # apply to noorinalabs/noorinalabs-data-acquisition
#   REPO=owner/name ./apply-ruleset.sh # override target repo (for re-use)
#   DRY_RUN=1 ./apply-ruleset.sh       # print the payload + planned action, no write

set -euo pipefail

REPO="${REPO:-noorinalabs/noorinalabs-data-acquisition}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$SCRIPT_DIR/ruleset-main.json"
RULESET_NAME="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$PAYLOAD")"

if [[ ! -f "$PAYLOAD" ]]; then
  echo "ERROR: ruleset payload not found at $PAYLOAD" >&2
  exit 1
fi

echo "Repo:    $REPO"
echo "Ruleset: $RULESET_NAME"
echo "Payload: $PAYLOAD"
echo

# Is there already a ruleset with this name? (idempotent re-apply / update.)
EXISTING_ID="$(gh api "repos/$REPO/rulesets" \
  --jq ".[] | select(.name == \"$RULESET_NAME\") | .id" 2>/dev/null || true)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN — would $( [[ -n "$EXISTING_ID" ]] && echo "UPDATE ruleset $EXISTING_ID" || echo "CREATE a new ruleset" ):"
  cat "$PAYLOAD"
  exit 0
fi

if [[ -n "$EXISTING_ID" ]]; then
  echo "Updating existing ruleset id $EXISTING_ID ..."
  gh api -X PUT "repos/$REPO/rulesets/$EXISTING_ID" --input "$PAYLOAD"
else
  echo "Creating new ruleset ..."
  gh api -X POST "repos/$REPO/rulesets" --input "$PAYLOAD"
fi

echo
echo "=== Read-back verification ==="
gh api "repos/$REPO/rulesets" \
  --jq ".[] | select(.name == \"$RULESET_NAME\") | {id, name, enforcement, target}"
echo
echo "Confirm the required-status-check contexts and bypass actor in the ruleset"
echo "detail (gh api repos/$REPO/rulesets/<id>) before declaring #322 met for this repo."
