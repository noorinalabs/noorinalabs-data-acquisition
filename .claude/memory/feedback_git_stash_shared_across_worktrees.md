---
name: feedback_git_stash_shared_across_worktrees
description: The stash is shared repo-wide across worktrees; `git stash push <path>` on a clean path creates NO entry, so a following `git stash pop` pops someone ELSE's stash. Never stash to A/B-test a fix — swap the file with `git show <ref>:<path>`.
metadata:
  type: feedback
---

**The git stash is a single repo-wide stack, shared by every worktree.** It is not per-worktree, per-branch, or per-agent.

Two facts compose into a live footgun:

1. `git stash push -- <path>` where `<path>` has **no unstaged changes** creates **no stash entry** and exits **0** (it prints "No local changes to save" — easy to miss under `-q`).
2. `git stash pop` then pops whatever is at `stash@{0}` — which may be a *months-old entry from another branch or another agent's session*.

Observed 2026-07-09 while red/green-testing da#356 (PR #363). `src/resolve/disambiguate.py` had already been **committed**, so:

```sh
git stash push -q src/resolve/disambiguate.py   # no-op, creates nothing, rc=0
uv run pytest …                                 # "RED" run — actually ran WITH the fix
git stash pop -q                                # popped a stranger's stash:
# Auto-merging src/parse/name_quality.py
# CONFLICT (content): Merge conflict in src/parse/name_quality.py
```

The popped entry was `stash@{0}: On I.Horvat/0000-resolve-integration-723: redundant da#253 name_quality staged copy`. It left `src/parse/name_quality.py` in a `UU` conflicted state — a file **owned by another engineer on another issue**. The "red" measurement was also worthless: nothing was stashed, so the fix was still active.

Recovery (the entry survives a conflicted pop — git prints "The stash entry is kept"):

```sh
git checkout HEAD -- <conflicted-path>
git stash list        # verify the stranger's entries are still there
```

## Rule

**Never use `git stash` to A/B-test a fix against a base ref.** Swap the file content instead — it touches nothing shared and cannot pop a stranger's work:

```sh
cp src/resolve/disambiguate.py /tmp/fixed.py
git show origin/main:src/resolve/disambiguate.py > src/resolve/disambiguate.py
ENVIRONMENT=test uv run pytest <test> -q          # true RED against the base ref
cp /tmp/fixed.py src/resolve/disambiguate.py
git diff --quiet HEAD -- src/resolve/disambiguate.py && echo "restored"   # always verify
```

Corollary of the same "silent zero" family as [[feedback_silent_zero_is_not_a_measurement]]: a stash push that saved nothing and a red run that proved nothing both return success. If a red run does not print the assertion you expected to see fail, the harness is lying to you — check that the base ref is actually in place. Related worktree/stash friction: [[feedback_structural_gate_commit_friction]].
