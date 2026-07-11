---
name: structural-ontology pre-commit gate commit friction
description: RETIRED by da#405 — structural index is now a gitignored build product; no commit-the-index, no staleness-check gate, no merge-driver. Emit with python3 if you need it locally. da#723/da#405.
metadata:
  type: feedback
---

**RETIRED (da#405, child rollout of main#939, 2026-07-10).** The friction below —
committing `ontology/structural/{llms.txt,code-graph.json}`, the `structural-ontology-staleness`
pre-commit hook, the `structural-ontology` CI workflow, and the union merge-driver — is GONE.
The index is now a **gitignored build product** (`.gitignore` → `ontology/structural/`),
regenerated on demand and never committed, so concurrent da PRs never conflict on it.
`scripts/structural_ontology.py` now has only `emit` (`check` + `register-merge-driver` removed).
Kept below because the source-line-position sensitivity explains WHY the index conflicted.

---

Committing a `src/**.py` change in this repo USED TO trip the `structural-ontology-staleness`
pre-commit hook (`python3 scripts/structural_ontology.py check`), which regenerated the
generated index (`ontology/structural/{llms.txt,code-graph.json}`) and failed on drift. Two
non-obvious traps cost real time on the #723 integration commit:

**Why:** the hook keeps `src/` line-accurate symbol positions in `llms.txt`; any source edit
shifts them, so the index must be regenerated and staged WITH the source change. Because
pre-commit stashes unstaged changes, a *partial* source commit always mismatches (the index
reflects the full tree, the hook sees only the staged subset) — stage ALL source edits together
or the gate can't be satisfied.

**How to apply (only if you regenerate the gitignored index locally):**
1. Regenerate with **`python3 scripts/structural_ontology.py emit`**, NOT `uv run python …`.
   The two produce DIFFERENT line numbers. There is no longer a `check` to confirm against —
   the file is gitignored, so a regenerated copy just sits on disk untracked.
2. An **untracked `.claude/worktrees/<x>/` dir** (a git worktree checkout living inside the
   repo) makes pre-commit report "Unstaged files detected" and its stash/un-stash conflicts
   with ruff-format auto-fixes → "Rolling back fixes", commit aborts with nothing wrong in your
   files. Ensure no unstaged *tracked* changes remain (`git diff --name-only` empty) and the
   conflict clears.
3. ruff-format drift: format with the **pinned** version (`uvx ruff@0.15.11 format …`, the
   `.pre-commit-config.yaml` rev), not local `uv run ruff` — see [[feedback_ruff_format_check_before_push]].

Related: [[feedback_ruff_parent_config_bleed]].
