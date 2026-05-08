# Parser-Fixture Coverage Audit — noorinalabs-data-acquisition

**Audit date:** 2026-05-07  
**Auditor:** Sofia Cardoso (P3W7 Tier-1 audit)  
**Wave:** P3W7  
**Meta-issue:** noorinalabs/noorinalabs-main#300  
**Charter ref:** `.claude/team/charter/hooks.md` § 5 Parser-Fixture Coverage Requirements  
**Head SHA verified:** `49d52802` (via `gh api repos/.../git/trees/49d52802?recursive=1`)

---

## State Classification

**data-acquisition is dispatcher-style (already-migrated / tier-2).**

`.claude/hooks/` directory is absent at head SHA `49d52802`. No local hook files exist in the committed tree. All PreToolUse hooks are delegated via `settings.json` to absolute paths pointing at the parent repo (`noorinalabs-main/.claude/hooks/`). This is the already-migrated pattern — equivalent to Kofi's design-system and Mateo's user-service findings this wave.

Tree evidence (full `.claude/` subtree at head SHA):
```
.claude/settings.json
.claude/skills/...   (11 skill files)
.claude/team/...     (charter, feedback_log, roster)
# No .claude/hooks/ directory or files
```

---

## Hook Inventory

Hook discovery path: `.claude/hooks/` (committed tree at head SHA)  
**Local hook files found: 0**

All hooks active in this repo's sessions are sourced from the parent:

| Hook (parent path) | Role | Parser-class |
|--------------------|------|-------------|
| `noorinalabs-main/.claude/hooks/validate_commit_identity.py` | PreToolUse — validates git commit `-c user.name`/`-c user.email` flags | YES |
| `noorinalabs-main/.claude/hooks/block_no_verify.py` | PreToolUse — blocks `--no-verify` | Minimal (command presence check) |
| `noorinalabs-main/.claude/hooks/block_git_config.py` | PreToolUse — blocks `git config` user changes | Minimal |
| `noorinalabs-main/.claude/hooks/auto_set_env_test.py` | PreToolUse — sets test env vars | Non-parser |
| `noorinalabs-main/.claude/hooks/validate_labels.py` | PreToolUse — validates issue/PR labels | YES |
| `noorinalabs-main/.claude/hooks/validate_pr_ci_status.py` | PreToolUse — validates PR CI state | YES |

(Source: `.claude/settings.json` `hooks.PreToolUse[*].command` fields at head SHA.)

---

## Classification

### Parser-class hooks active in this child's sessions

**`validate_commit_identity.py` (parent)** — Parser-class YES. Full shlex-tokenized parser as of P3W6 fix. Covered by parent fixtures at `noorinalabs-main/.claude/hooks/tests/test_validate_commit_identity.py`. Those fixtures cover the parent code path — which IS the code path running in this child's sessions.

**`validate_labels.py` (parent)** — Parser-class YES (parses `gh issue create`/`gh pr create` argument lists to extract `--label` values). Coverage: deferred to parent-repo audit scope.

**`validate_pr_ci_status.py` (parent)** — Parser-class YES (parses `gh pr merge` command and PR CI state). Coverage: deferred to parent-repo audit scope.

### Data-acquisition-specific input shapes not covered by parent fixtures

The parent fixtures are written against generic roster and command inputs. This child repo has domain-specific input patterns that no parent fixture currently exercises:

| Input shape | Relevant hook | Fixture gap |
|-------------|--------------|------------|
| `cd /path/to/noorinalabs-data-acquisition && git commit ...` cross-repo commit | `validate_commit_identity` | Parent fixtures use `tempfile` synthetic repos; no fixture uses the real data-acquisition path shape with its merged roster (Dilara, Alejandra, etc.) |
| `gh issue create --repo noorinalabs/noorinalabs-data-acquisition --label "tech-debt,p3-wave-7"` | `validate_labels` | No data-acquisition-specific label combination fixture |
| `gh pr merge` against `deployments/phase-3/wave-7` head | `validate_pr_ci_status` | No wave-branch-specific input shape fixture |

---

## Coverage Table

| Hook (parent path) | Parser-class | Parent fixture coverage | Data-acq-specific fixture | Gap |
|--------------------|-------------|------------------------|--------------------------|-----|
| `validate_commit_identity.py` | YES | YES (comprehensive) | NO — no fixture exercises real data-acq roster or path | Low-medium: synthetic fixtures cover the code paths; real-roster shape is a documentation gap |
| `validate_labels.py` | YES | Partial | NO | Medium: label set distinct from parent |
| `validate_pr_ci_status.py` | YES | Partial | NO | Low: wave-branch shape not exercised |
| `block_no_verify.py` | Minimal | YES | N/A | None |
| `block_git_config.py` | Minimal | YES | N/A | None |
| `auto_set_env_test.py` | Non-parser | N/A | N/A | None |

---

## Summary

**0 local hooks, 0 local fixture files. data-acquisition is already dispatcher-style.**

The repo delegates entirely to parent hooks — the correct pattern for a child repo that does not need specialized hook behavior. The parent's `validate_commit_identity.py` is the current shlex-tokenized version; its fixture suite covers the parser code paths that run in this child's sessions.

Residual gap: no fixture exercises data-acquisition-specific input shapes (real roster names, data-acq label combinations, wave-7 branch head patterns). This is a parent-test augmentation gap, not a local-file gap.

---

## Pattern G Observations

### G1: Already dispatcher (correctly migrated)

data-acquisition has no `.claude/hooks/` directory — it is pure dispatcher-style. `settings.json` delegates to parent absolute paths. This is the correct architecture for child repos without specialized hook requirements. No action needed on local hook files (there are none).

**Note on initial audit error:** An earlier draft of this audit incorrectly described local hook files (`validate_commit_identity.py`, `annunaki_log.py`) as "stale pre-shlex dead code." Those files are present in the parent repo's main working tree and were visible during local filesystem enumeration, but they are NOT committed to the child repo's git tree. This audit now reflects the verified committed state at head SHA.

### G2: No child-specific fixture augmentation in parent

The parent fixture suite at `noorinalabs-main/.claude/hooks/tests/` does not include data-acquisition-specific shapes. If a child-specific parser bug surfaces (e.g., a commit from "Dilara Erdogan" with a path containing data-acquisition-specific characters, or a label combination unique to this repo), no fixture pins that shape.

**Recommendation:** Parent-test augmentation — add data-acquisition-specific roster + label fixtures to the parent's test suite as a backport item.

---

## Gap Issues

| Issue | Title | Disposition |
|-------|-------|------------|
| [#43](https://github.com/noorinalabs/noorinalabs-data-acquisition/issues/43) | Re-scoped: augment parent fixture suite with data-acquisition-specific input shapes | Active — re-scoped body |
| [#44](https://github.com/noorinalabs/noorinalabs-data-acquisition/issues/44) | Re-scoped: architectural decision record — does data-acq ever need local hooks? | Active — re-scoped body (no files to remove; decision record for future) |
