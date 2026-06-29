---
name: structural-ontology pre-commit gate commit friction
description: regenerate the structural index with python3 (NOT uv run) to match the hook; an untracked .claude/worktrees dir breaks pre-commit's stash. da#723.
metadata:
  type: feedback
---

Committing a `src/**.py` change in this repo trips the `structural-ontology-staleness`
pre-commit hook (`python3 scripts/structural_ontology.py check`), which regenerates the
generated index (`ontology/structural/{llms.txt,code-graph.json}`) and fails on drift. Two
non-obvious traps cost real time on the #723 integration commit:

**Why:** the hook keeps `src/` line-accurate symbol positions in `llms.txt`; any source edit
shifts them, so the index must be regenerated and staged WITH the source change. Because
pre-commit stashes unstaged changes, a *partial* source commit always mismatches (the index
reflects the full tree, the hook sees only the staged subset) — stage ALL source edits together
or the gate can't be satisfied.

**How to apply:**
1. Regenerate with **`python3 scripts/structural_ontology.py emit`**, NOT `uv run python …`.
   The two produce DIFFERENT line numbers (the hook runs bare `python3`); a `uv run`-generated
   index fails the `python3`-run `check` even though both are ruff/mypy-clean. Confirm with
   `python3 scripts/structural_ontology.py check` → "OK: …current".
2. An **untracked `.claude/worktrees/<x>/` dir** (a git worktree checkout living inside the
   repo) makes pre-commit report "Unstaged files detected" and its stash/un-stash conflicts
   with ruff-format auto-fixes → "Rolling back fixes", commit aborts with nothing wrong in your
   files. Ensure no unstaged *tracked* changes remain (`git diff --name-only` empty) and the
   conflict clears.
3. ruff-format drift: format with the **pinned** version (`uvx ruff@0.15.11 format …`, the
   `.pre-commit-config.yaml` rev), not local `uv run ruff` — see [[feedback_ruff_format_check_before_push]].

Related: [[feedback_ruff_parent_config_bleed]].
