# ADR-001: Local hook policy — data-acquisition stays dispatcher-style

## Status: Accepted (P3W8)

## Context

The Phase-3 Wave-7 parser-fixture coverage audit (data-acquisition#45, parent meta-issue #300)
confirmed that `noorinalabs-data-acquisition` has **no committed `.claude/hooks/` files**. All
Claude Code automation hooks are delegated to the parent canonical at
`noorinalabs-main/.claude/hooks/` via absolute paths in this repo's `.claude/settings.json`.

The org-level charter recognizes a *dispatcher-style* exemption category for children in
this topology — see `charter/hooks.md § Parser-Fixture Coverage Requirements —
dispatcher-children sub-clause` (pending noorinalabs/noorinalabs-main#311). The W7 audit
(parent#300) and Aino's W8 charter edit (parent#311) name data-acquisition as one of the
canonical dispatcher-style exemplars. This ADR ratifies the local effect of that
charter rule for `noorinalabs-data-acquisition`; it does not redefine the rule.

Issue #44 was filed in P3W7 originally framed as "remove dead-code child hook copies",
then re-scoped during the W7 audit when committed-tree inspection confirmed there was
nothing to remove. The re-scoped framing asks the architectural question instead: should
`noorinalabs-data-acquisition` ever introduce local hooks, and if so, what is the
onboarding pattern?

### Verification at HEAD

Applying the charter's dispatcher-style classification check (defined in parent#311) to
this repo at the wave-8 base SHA `f43b84d19d1e38f4b964c6ed313ce5cc58584914`:

```bash
gh api repos/noorinalabs/noorinalabs-data-acquisition/git/trees/f43b84d19d1e38f4b964c6ed313ce5cc58584914?recursive=1 \
  --jq '.tree[] | select(.path|startswith(".claude/hooks/")) | .path'
# (no output — 0 entries under .claude/hooks/ → dispatcher-style per charter)
```

`.claude/settings.json` references the parent canonical via absolute paths, e.g.:

```
python3 /home/parameterization/code/noorinalabs-main/.claude/hooks/dispatcher.py
python3 /home/parameterization/code/noorinalabs-main/.claude/hooks/enforce_librarian_consulted.py
python3 /home/parameterization/code/noorinalabs-main/.claude/hooks/validate_wave_context.py
```

There are no `.py` files committed under `.claude/hooks/` for the parser-fixture
requirement to apply to.

## Decision

**`noorinalabs-data-acquisition` remains dispatcher-style.** All Claude Code hook
execution is delegated to the parent canonical at `noorinalabs-main/.claude/hooks/`. No
local hook files are committed to this repository. Coverage obligations under
`charter/hooks.md § Parser-Fixture Coverage Requirements` are fulfilled by the parent's
test suite per the dispatcher-children sub-clause (pending parent#311 — citation will
resolve to merged charter language).

The door is explicitly left open to revisit this decision if a concrete domain-specific
hook need emerges that the parent does not naturally cover.

## Rationale

- **Single source of truth.** Hook logic lives in one place; bug fixes and behavior changes
  apply uniformly across all dispatcher-style children without per-repo backport churn.
- **Smaller repo footprint.** No duplicated hook code, no duplicated fixtures, no
  per-child test runner configuration to maintain.
- **Parent test suite is authoritative.** The parent's `tests/test_*.py` cover all hook
  behavior in one place; W7 audit (#45) and parent#311 codify this as the coverage
  contract for dispatcher-style children.
- **Aligns with the rest of the org.** Six of seven children are dispatcher-style; the
  pattern is the established norm rather than an exception.
- **No present need.** No data-acquisition-specific gate has been identified that the
  parent canonical hooks cannot already satisfy. The team has not surfaced a hook
  requirement (e.g., scraper-manifest validation, B2-path checks, Kafka topic name
  enforcement) that warrants its own implementation.

## Consequences

### Positive

- This repo's PRs do not need to ship per-repo hook tests; CI in `noorinalabs-main`
  carries the load.
- Onboarding new agents in this repo requires no local hook setup beyond the parent's
  shared `.claude/settings.json` references.
- Behavior is consistent with the rest of the org — agents working across repos see the
  same hook surface everywhere.

### Negative / accepted trade-offs

- Per-repo customization of hook behavior is not possible without revisiting this ADR.
  If data-acquisition develops a domain-specific gate need, the team must either:
  1. **Revisit this ADR** (file a superseding ADR), add a `.claude/hooks/` directory with
     local files, and meet the per-child fixture requirement in
     `charter/hooks.md § Parser-Fixture Coverage Requirements`; or
  2. **Propose a parent-side change** — extend the canonical hook in
     `noorinalabs-main/.claude/hooks/` with logic that conditionally activates for
     data-acquisition (e.g., via cwd-keyed dispatch).

  Option 2 should be preferred whenever the gate logic is generalizable across repos.
  Option 1 is appropriate only when the surface is genuinely data-acquisition-specific
  (e.g., a scraper-only command shape that no other repo would meaningfully consume).

- Changes to parent canonical hooks affect this repo immediately on the next session.
  The team relies on parent's CI gating and the parent#311 dispatcher-children contract
  to keep that behavior stable.

### Trigger conditions for revisiting

This ADR should be revisited if any of the following surface:

- A scraper-only or acquire-only command shape that the parent dispatcher cannot
  reasonably parse without leaking data-acquisition-specific knowledge into the parent.
- A B2 / Kafka / staging-path gate that is so domain-specific that putting it in the
  parent would muddy the parent's separation of concerns.
- A team decision (recorded in a follow-up issue) that local hook ownership is needed
  for autonomy reasons unrelated to a specific gate.

In any of those cases: file a superseding ADR (e.g., `ADR-00X: Local hooks for
<purpose>`), update this file's status to `Superseded by ADR-00X`, and follow the
per-child fixture obligations in `charter/hooks.md § Parser-Fixture Coverage Requirements`
for the new local hooks.

## References

- Issue: noorinalabs/noorinalabs-data-acquisition#44 (this ADR closes it)
- W7 audit (data-acquisition exemplar): noorinalabs/noorinalabs-data-acquisition#45 (merged)
- Parent meta-issue (W7 parser-fixture audit): noorinalabs/noorinalabs-main#300 (closed)
- Parent charter sub-clause introducing dispatcher-style exemption:
  noorinalabs/noorinalabs-main#311
- Parent charter file (canonical hook policy):
  `noorinalabs-main/.claude/team/charter/hooks.md § Parser-Fixture Coverage Requirements`
- Org-level conventions (hook table):
  `noorinalabs-main/ontology/conventions.md § Automation hooks (org-level)`
