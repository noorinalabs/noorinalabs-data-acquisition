# Branch Protection — noorinalabs-data-acquisition (P3 end-state #4, main#322)

Phase-3 end-state criterion #4 (`noorinalabs-main#322`): **CI failures block all
merges** on every repo's default branch, org-wide — enforced server-side by
GitHub, not only by the Hook 4 comment-gate. This directory carries the
canonical ruleset documentation for this repo's `main`:

| File | Purpose |
|------|---------|
| `ruleset-main.json` | The repository ruleset payload (GitHub REST `/rulesets`) — documents the LIVE ruleset. |
| `apply-ruleset.sh`  | Owner/admin-gated create-or-update + read-back-verify. Idempotent. |
| `SPEC.md`           | This document — the shape and the why. |

This is data-acquisition's adoption of the parent-canonical spec
(`noorinalabs-main` charter `pull-requests.md` § *Org-Wide Branch Protection +
Admin-Merge Exceptions*).

## Application status — ALREADY APPLIED (W13 pilot)

> **data-acquisition is the W13 `#322` branch-protection PILOT.** The ruleset
> below is **already live** on this repo's `main` (ruleset id **`17091263`**,
> created `2026-05-31`, `enforcement: active`). These files **document** the
> live ruleset for parity with the other repos rolled out in W14 — they do NOT
> introduce a new apply. The owner has already created and read-back-verified
> the protection; `apply-ruleset.sh` here is idempotent and would `PUT`-update
> the existing same-named ruleset rather than create a duplicate, but there is
> **no need to run it** unless the documented shape below intentionally diverges
> from the live ruleset.

So #322 is **met for this repo** (the protection is live). `#322` stays OPEN in
`noorinalabs-main` as the org-wide rollout tracker until all 8 default branches
carry the protection — data-acquisition's box is already checked.

The `ruleset-main.json` in this directory was reconciled against the live
ruleset at W14 PR time (`gh api repos/noorinalabs/noorinalabs-data-acquisition/rulesets/17091263`)
and matches it field-for-field.

## The ruleset shape (and why)

A **repository ruleset** targeting `~DEFAULT_BRANCH`, `enforcement: active`:

- **`pull_request` with `required_approving_review_count: 0`** — the load-bearing
  decision. GitHub's "require approvals" counts **formal** GitHub PR reviews,
  which our team structurally cannot produce: the `gh` auth principal IS the PR
  author (`parametrization`), so a formal self-approval **422s**, and our review
  discipline runs on **issue-comment verdicts** validated by Hook 4
  (`validate_pr_review`), not formal reviews. A naive "require 1 approval" rule
  would **deadlock every merge**. Reviewer-count enforcement stays with Hook 4.
- **`required_status_checks` (strict)** — data-acquisition has **unconditional PR
  CI** (no `paths:` filter on `ci.yml`), so the ruleset hard-requires its gate
  **job-name** contexts:

  | Context | Source job |
  |---------|-----------|
  | `Lint` | `ci.yml` → `lint` (ruff check + ruff format check) |
  | `Type Check` | `ci.yml` → `typecheck` (mypy) |
  | `Test` | `ci.yml` → `test` (pytest, unit) |
  | `Integration Tests` | `ci.yml` → `test-integration` (pytest, integration) |

  These match the contexts in the live ruleset (id `17091263`). The
  `Pre-commit ⇄ CI sync-drift gate` job added in W14 is an additional
  unconditional gate; it is **intentionally not** in the required-contexts set
  here (it gates mirror-drift, a soft-fail-direction process check, and the
  owner can add `{ "context": "Pre-commit ⇄ CI sync-drift gate" }` to the live
  ruleset later if desired). The `Integration Tests` job is `continue-on-error`
  in `ci.yml`, so it reports a passing check-run even when integration tests
  fail — its presence as a required context gates job COMPLETION, not the
  integration result. **Re-confirm all contexts at any future apply** against
  live check-runs — job names can change:
  `gh api repos/<repo>/commits/<default-sha>/check-runs --jq '.check_runs[].name'`.
- **`deletion` + `non_fast_forward`** — no force-push / branch-delete on `main`.
- **`bypass_actors`: Repository-admin (`actor_id: 5`, `bypass_mode: always`)** —
  keeps the orchestrator's `--admin` wave→main wrapup merges and the charter
  single-reviewer / doc-sweep / emergency exceptions working. The GitHub-side
  bypass is mirrored on the operator side by the hook-validated
  `ADMIN_MERGE_EXCEPTION` gate (`validate_pr_ci_status`), which **audits** every
  `--admin` merge to the Annunaki trail — defense in depth: the ruleset covers
  UI/external/batch-loop merges, the hook covers `gh pr merge` and names the
  exceptions.

## How to re-verify or re-apply (owner)

The ruleset is already live; these commands are for re-verification or for a
deliberate update if the documented shape changes:

```bash
# Read-back-verify the live ruleset detail (contexts + bypass actor):
gh api repos/noorinalabs/noorinalabs-data-acquisition/rulesets \
  --jq '.[] | select(.name|startswith("Protect main")) | .id'
gh api repos/noorinalabs/noorinalabs-data-acquisition/rulesets/17091263

# Only if the documented shape intentionally diverges and must be pushed:
# (idempotent — PUT-updates the existing same-named ruleset, no duplicate)
.github/branch-protection/apply-ruleset.sh            # create or update
DRY_RUN=1 .github/branch-protection/apply-ruleset.sh  # preview only
```
