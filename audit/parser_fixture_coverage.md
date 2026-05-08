# Parser-Fixture Coverage Audit — noorinalabs-data-acquisition

**Audit date:** 2026-05-07  
**Auditor:** Sofia Cardoso (P3W7 Tier-1 audit)  
**Wave:** P3W7  
**Meta-issue:** noorinalabs/noorinalabs-main#300  
**Charter ref:** `.claude/team/charter/hooks.md` § 5 Parser-Fixture Coverage Requirements

---

## Hook Inventory

Hook discovery path: `.claude/hooks/`  
Total hooks found: **2**

| # | File | Role |
|---|------|------|
| 1 | `validate_commit_identity.py` | PreToolUse — validates git commit `-c user.name`/`-c user.email` flags against roster |
| 2 | `annunaki_log.py` | Shared utility — JSONL log writer called by blocking hooks; not a hook itself |

---

## Classification

### Hook 1: `validate_commit_identity.py`

**Parser-class: YES**

This hook contains substantial input parsing:

- **Command detection** — `_is_git_commit_command()` uses regex to find `git ... commit` at command position; strips heredocs and quoted strings first.
- **Flag extraction** — regex-based extraction of `-c user.name=` and `-c user.email=` values, including handling quoted vs unquoted values.
- **Cross-repo detection** — `_detect_target_roster()` parses `cd <path>` out of the command string.
- **Roster loading** — JSON parsing of `roster.json` files; merge logic for parent+child repos.

**Known parser issues fixed in parent repo (not yet backported to child):**

The child-repo copy differs from the parent repo's current `validate_commit_identity.py`:

| Aspect | Child-repo version | Parent-repo current version |
|--------|--------------------|------------------------------|
| Tokenization | Regex-only (`re.search` on raw string) | `shlex`-based tokenization via `_shell_parse` module |
| Flag extraction | `re.search(r'-c\s+user\.name=...')` on full command | `extract_dash_c_pairs()` from tokenized segment |
| Parse-failure handling | Not present — regex always returns a result | Fail-closed: shlex failure on commit-shaped command blocks |
| Input Language docblock | Absent | Present |
| Repeated `-c user.name` | Undefined behavior (first match wins) | Last-value-wins (matches git semantics) |

**Bug class**: The child copy exhibits the pre-P3W6 regex parser bugs that the parent fixed: unquoted `-c user.email=val` could slurp to EOL; nested heredoc-in-command-sub could mangle the parser. The parent version addresses these via `#226`, `#188`, and `#287`.

### Hook 2: `annunaki_log.py`

**Parser-class: NO** (utility/writer, not a hook)

`annunaki_log.py` is a shared logging utility — it writes structured JSONL records. It does not parse input from the harness; it receives structured Python arguments from calling hooks. No input parsing requiring fixtures.

The child copy also diverges from the parent:
- Uses `datetime.now(timezone.utc)` instead of `datetime.now(UTC)` (cosmetic)
- Lacks `append_jsonl_record()` helper (added in parent for writer hardening)
- Raw `open()` write instead of the hardened append helper

These are sync gaps (not parser bugs), tracked as a backport concern.

---

## Coverage Table

| Hook | Parser-class | Has local fixture tests | Parent fixtures exist | Gap severity |
|------|-------------|------------------------|----------------------|-------------|
| `validate_commit_identity.py` | YES | NO | YES (`.claude/hooks/tests/test_validate_commit_identity.py` in parent) | **HIGH** — child copy is a stale regex variant not covered by parent fixtures (different code path) |
| `annunaki_log.py` | NO | N/A | N/A | None (utility, not parser-class) |

---

## Summary

**1 parser-class hook, 0 fixture files in child repo.**

The child repo's `validate_commit_identity.py` is a stale copy of the parent's hook — predating the `shlex` tokenization refactor that fixed multiple parser bugs (#226, #188, #287). The parent repo has a comprehensive fixture test suite at `.claude/hooks/tests/test_validate_commit_identity.py`, but:

1. Those tests import the parent's hook (with `_shell_parse` module), not the child's regex-only copy.
2. The child's `.claude/hooks/` has no test directory or fixture files.
3. The child's hook is the one actually invoked by `settings.json` references? — **No**: `settings.json` delegates to the parent's `validate_commit_identity.py` directly. The child's local copy is **dead code** — it is not registered anywhere and not invoked by the harness.

---

## Pattern G Observations

### G1: Dead code risk

The child-repo copies of `validate_commit_identity.py` and `annunaki_log.py` are not referenced in `settings.json` (which calls `python3 /home/parameterization/code/noorinalabs-main/.claude/hooks/validate_commit_identity.py` — the parent path). These files create a false impression of local coverage while being silently inert. They diverge from the parent and will continue to drift.

**Recommendation:** Either delete the child copies (settings.json already delegates to parent) or add a sync-check test that fails if child and parent diverge.

### G2: No `tests/` under `.claude/hooks/`

Even for documentation/clarity, zero fixture files exist under `.claude/hooks/tests/`. The charter (§ 5) requires fixture coverage for parser-class hooks. If the child ever registers its own hooks (e.g., a data-source-manifest validator for B2 paths or Kafka topic names), there will be no test infrastructure ready.

---

## Gap Issues Filed

See backport issues below for tracking.

| Issue | Title | Labels |
|-------|-------|--------|
| [#43](https://github.com/noorinalabs/noorinalabs-data-acquisition/issues/43) | tech-debt(hooks): sync validate_commit_identity.py + annunaki_log.py with parent shlex-tokenized versions + add fixture tests | `tech-debt`, `p3-wave-7` |
| [#44](https://github.com/noorinalabs/noorinalabs-data-acquisition/issues/44) | cleanup(hooks): remove or align dead-code child hook copies (not registered in settings.json) | `tech-debt`, `p3-wave-7` |
