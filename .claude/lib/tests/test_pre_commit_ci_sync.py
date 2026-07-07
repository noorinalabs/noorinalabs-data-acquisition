"""Tests for pre_commit_ci_sync — the pre-commit <-> CI drift gate (#327).

Focused on the noorinalabs-main#684 cspell rollout for this repo: the CI
`Spellcheck (cspell)` job (.github/workflows/docs.yml) must classify as the
`cspell` kind, a pre-commit cspell hook must classify too, and the real repo
config must mirror the COMPLETE CI check-set across ALL workflows (no harmful
drift).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Helper lives at .claude/lib/pre_commit_ci_sync.py; this test is at
# .claude/lib/tests/test_*.py. parent.parent reaches the lib root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pre_commit_ci_sync import (  # noqa: E402
    check_repo,
    compute_drift,
    kinds_from_ci,
    kinds_from_precommit,
)

# .claude/lib/tests/test_*.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class CspellKindClassification(unittest.TestCase):
    """#684: the spell-check blind spot. A CI Spellcheck job must classify as
    the `cspell` kind on EITHER expression — the `streetsidesoftware/cspell-action`
    `uses:` ref, a bundled-CLI `cspell` step/run, or the generic `spellcheck`
    word — and a pre-commit cspell hook must classify too, so a CI spell gate
    with no local mirror produces harmful drift instead of silence."""

    def test_cspell_action_uses_ref_classified(self) -> None:
        wf = """
jobs:
  spellcheck:
    name: Spellcheck (cspell)
    steps:
      - name: cspell
        uses: streetsidesoftware/cspell-action@de2a73e # v8.4.0
"""
        self.assertIn("cspell", kinds_from_ci(wf))

    def test_cspell_cli_run_step_classified(self) -> None:
        wf = """
jobs:
  spell:
    steps:
      - run: npx cspell --config .cspell.json "**/*.md"
"""
        self.assertIn("cspell", kinds_from_ci(wf))

    def test_generic_spellcheck_word_classified(self) -> None:
        # A repo that names the step/run with the generic word still registers.
        self.assertIn("cspell", kinds_from_ci("      - run: make spellcheck\n"))

    def test_precommit_cspell_hook_classified(self) -> None:
        cfg = """
repos:
  - repo: https://github.com/streetsidesoftware/cspell-cli
    rev: v8.4.0
    hooks:
      - id: cspell
        name: cspell
"""
        self.assertIn("cspell", kinds_from_precommit(cfg))

    def test_ci_cspell_without_precommit_is_harmful_drift(self) -> None:
        # The exact divergence #684 exists to catch: CI enforces cspell, the
        # pre-commit config does not mirror it.
        wf = """
jobs:
  spellcheck:
    steps:
      - uses: streetsidesoftware/cspell-action@de2a73e
"""
        cfg = """
repos:
  - repo: local
    hooks:
      - id: ruff
"""
        harmful, _ = compute_drift(kinds_from_precommit(cfg), kinds_from_ci(wf))
        self.assertIn("cspell", harmful)

    def test_ci_cspell_with_precommit_mirror_no_drift(self) -> None:
        wf = """
jobs:
  spellcheck:
    steps:
      - uses: streetsidesoftware/cspell-action@de2a73e
"""
        cfg = """
repos:
  - repo: https://github.com/streetsidesoftware/cspell-cli
    rev: v8.4.0
    hooks:
      - id: cspell
"""
        harmful, _ = compute_drift(kinds_from_precommit(cfg), kinds_from_ci(wf))
        self.assertNotIn("cspell", harmful)


class RealRepoCspellParity(unittest.TestCase):
    """Guards on this repo's actual committed config — the rollout must stay
    enforced, not just pass on synthetic fixtures."""

    def _workflow_paths(self) -> list[Path]:
        wf_dir = _REPO_ROOT / ".github" / "workflows"
        return sorted(p for p in wf_dir.glob("*.y*ml") if p.is_file())

    def test_ci_enforces_cspell(self) -> None:
        # Guard against the docs.yml cspell job being removed/renamed without the
        # mirror+kind being revisited: the kind must actually appear in CI.
        ci_kinds: set[str] = set()
        for path in self._workflow_paths():
            ci_kinds |= kinds_from_ci(path.read_text(encoding="utf-8"))
        self.assertIn("cspell", ci_kinds)

    def test_precommit_mirrors_all_workflows(self) -> None:
        # The CI gate globs ALL workflow files (#684), so this repo must have no
        # harmful drift across the full set — in particular docs.yml carries the
        # cspell + actionlint jobs. This is the gate exactly as the
        # `Pre-commit ⇄ CI sync-drift gate` CI job runs it.
        precommit = _REPO_ROOT / ".pre-commit-config.yaml"
        self.assertTrue(precommit.is_file(), "repo must have a pre-commit config")
        ci_paths = self._workflow_paths()
        self.assertTrue(ci_paths, "repo must have workflow files")
        harmful, _ = check_repo(precommit, ci_paths)
        self.assertEqual(
            harmful,
            set(),
            f"pre-commit must mirror ALL CI workflows; missing locally: {sorted(harmful)}",
        )


class DockerBuildxNotBuildKind(unittest.TestCase):
    """Regression for the `build`-kind false-match (deploy#391 / isnad-graph#938,
    da#329): a docker image-PUBLISH workflow uses `Set up Docker Buildx` and
    `docker buildx build`, whose substring `docker build` used to classify as the
    `build` quality-gate kind. A registry publish is not a local fast-feedback
    check, so that produced a permanent un-mirrorable `build` drift and the gate
    could never exit 0 on a docker-publish repo. The pattern now matches only
    genuine build-quality-gate names (`build-and-validate`, `build-and-test`,
    `npm run build`)."""

    def test_buildx_setup_step_name_not_build_kind(self) -> None:
        wf = """
jobs:
  build-and-push-load:
    steps:
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@d7f5e7f # v4
"""
        self.assertNotIn("build", kinds_from_ci(wf))

    def test_buildx_build_run_line_not_build_kind(self) -> None:
        self.assertNotIn("build", kinds_from_ci("      - run: docker buildx build --push .\n"))

    def test_publish_workflow_no_harmful_build_drift(self) -> None:
        # The whole point: a repo whose only build-ish CI is a docker publish
        # (no npm/compile quality gate) must not see harmful `build` drift.
        wf = """
jobs:
  build-and-push-load:
    steps:
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@d7f5e7f # v4
      - name: Build and push
        uses: docker/build-push-action@f9f3042 # v7
"""
        cfg = """
repos:
  - repo: local
    hooks:
      - id: ruff
"""
        harmful, _ = compute_drift(kinds_from_precommit(cfg), kinds_from_ci(wf))
        self.assertNotIn("build", harmful)

    def test_real_build_quality_gate_still_classified(self) -> None:
        # A genuine build-quality gate must still register so it stays mirrored.
        self.assertIn("build", kinds_from_ci("      - run: npm run build\n"))


if __name__ == "__main__":
    unittest.main()
