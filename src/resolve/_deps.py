"""Declared-dependency guards for resolve stages (da#309).

A resolve stage may be *skipped* for exactly one legitimate reason: the operator
disabled it. A stage that is **enabled** but whose **declared** third-party
dependencies are absent is an *environment defect*, not an empty result — and it
must never be reported as a successful run that simply found nothing.

Before da#309, :mod:`src.resolve.dedup` caught ``ImportError`` on
``sentence-transformers`` / ``faiss`` and returned a schema-valid, zero-row
``parallel_links.parquet``. Every downstream consumer accepted that table as a
genuine "no parallels found", so a pipeline run on a box without the ``ml``
dependency group silently produced **zero** ``PARALLEL`` edges (production
carries ~4.49M) and still reported success.

The two conditions are now distinct *in code* rather than by log level:

* **disabled stage** — the caller passes ``require=False`` (or sets the
  corresponding setting), which is an explicit, auditable choice to run the
  no-model fallback. The stage logs a named degraded-skip event.
* **enabled stage, dependency absent** — :func:`require_dependencies` raises
  :class:`MissingDependencyError`.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence

from src.exit_codes import EXIT_MISSING_DEPENDENCY

__all__ = [
    "EXIT_MISSING_DEPENDENCY",
    "MissingDependencyError",
    "missing_dependencies",
    "require_dependencies",
]


class MissingDependencyError(BaseException):
    """An enabled resolve stage is missing one or more declared dependencies.

    Subclasses ``BaseException`` (not ``Exception``) **on purpose**, for the same
    reason as :class:`~src.resolve._checkpoint.StopAfterReached`: the resolve
    orchestrator wraps each stage in ``except Exception`` and logs-and-continues,
    so an ``Exception`` raised here would be swallowed as a "step failed" and the
    pipeline would march on to the next stage and report
    ``resolve_pipeline_complete``. That is precisely the silent success this guard
    exists to prevent, so the error is raised as a ``BaseException`` to sail through
    those guards and surface to the CLI, which maps it to
    :data:`EXIT_MISSING_DEPENDENCY`.

    The message names *which* stage, *which* dependencies, and *what to run*, so an
    operator can act on it without reading the traceback.
    """

    def __init__(
        self,
        *,
        stage: str,
        missing: Sequence[str],
        dependency_group: str,
        remediation: str,
    ) -> None:
        self.stage = stage
        self.missing = list(missing)
        self.dependency_group = dependency_group
        self.remediation = remediation
        super().__init__(self._render())

    def _render(self) -> str:
        deps = ", ".join(self.missing)
        plural = "dependency" if len(self.missing) == 1 else "dependencies"
        return (
            f"resolve stage {self.stage!r} is enabled but its declared {plural} "
            f"({deps}) could not be imported. This is an environment defect, not an "
            f"empty result: continuing would write a zero-row output that is "
            f"indistinguishable from a genuine negative. "
            f"Install the {self.dependency_group!r} dependency group with: {self.remediation}"
        )


def missing_dependencies(modules: Iterable[str]) -> list[str]:
    """Return the subset of ``modules`` that cannot be imported, preserving order."""
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return missing


def require_dependencies(
    *,
    stage: str,
    modules: Iterable[str],
    dependency_group: str,
    remediation: str,
) -> None:
    """Raise :class:`MissingDependencyError` if any of ``modules`` is unimportable.

    Call this at the *entry* of an enabled stage, before any input is read: an
    enabled stage whose declared dependency is absent is a defect regardless of
    whether its input happens to be empty.
    """
    missing = missing_dependencies(modules)
    if missing:
        raise MissingDependencyError(
            stage=stage,
            missing=missing,
            dependency_group=dependency_group,
            remediation=remediation,
        )
