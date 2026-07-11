"""Required-input guards for resolve stages (da#361).

Companion to :mod:`src.resolve._deps` (da#309), one layer up. ``_deps`` guards a
missing *dependency* (an unimportable module); this guards a missing *input*
artifact (a parquet an upstream stage should have produced).

The defect class (da#361)
-------------------------
Several ``src/resolve`` stages returned a success-shaped empty result — ``[]`` /
``None`` / a zero-row parquet — when a **required** input artifact was absent. A
missing input means an upstream stage did not run (or ran against the wrong
directory): an upstream defect. It is NOT the same thing as an input that ran and
legitimately produced nothing, yet the two were indistinguishable to every caller
and to the process exit status — the same silent success #309 fixed for
dependencies and #360 fixed for a swallowed raise.

Distinguishing the two is the whole point, so this guard fires **only on
absence**, never on a present-but-empty input. Each call site keeps its honest
empty return for the true-negative case (input present, genuinely zero rows) and
delegates only the *absence* decision here.

Why ``Exception``, not ``BaseException``
----------------------------------------
:class:`~src.resolve._deps.MissingDependencyError` subclasses ``BaseException`` so
it **sails past** ``run_all``'s per-stage ``except Exception`` and maps to its own
exit code (``4``). :class:`MissingInputError` makes the opposite choice **on
purpose**: a missing input is exactly the "a stage could not do its job"
condition da#360's machinery already models. A plain ``Exception`` is *caught* by
``run_all`` and recorded as a :class:`~src.resolve.StageErrored`, which makes
``run_all`` raise :class:`~src.resolve.ResolveStageError` at the end → exit
:data:`~src.exit_codes.EXIT_STAGE_FAILED` (11). So a stage that raises on a
missing input is aggregated with every other failed stage rather than aborting
the best-effort sweep, and needs no new exit code.

da#360 is the reason raising is meaningful at all here: before it, ``run_all``
swallowed the raise, logged ``resolve_step_failed`` and marched on to
``resolve_pipeline_complete``, exiting 0. Raising a missing-input error into that
old orchestrator would have been swallowed exactly like the empty return it
replaces — which is precisely why da#361's "Proposed direction" made this depend
on da#360.
"""

from __future__ import annotations

__all__ = ["MissingInputError", "require_input"]


class MissingInputError(Exception):
    """An enabled resolve stage's **required** input artifact is absent.

    Distinct from a genuinely empty input (input present, zero rows), which each
    stage still reports as an honest success. The message names *which* stage,
    *which* input, *which* upstream stage should have produced it, and *what to
    run* — so an operator can act without reading the traceback.

    A plain ``Exception`` (not ``BaseException``) so ``run_all`` catches it as a
    :class:`~src.resolve.StageErrored`; see this module's docstring for why.
    """

    def __init__(
        self,
        *,
        stage: str,
        input_desc: str,
        produced_by: str,
        remediation: str,
    ) -> None:
        self.stage = stage
        self.input_desc = input_desc
        self.produced_by = produced_by
        self.remediation = remediation
        super().__init__(self._render())

    def _render(self) -> str:
        return (
            f"resolve stage {self.stage!r} is missing its required input "
            f"({self.input_desc}), normally produced by {self.produced_by}. This is an "
            f"upstream defect, not an empty result: continuing would write a "
            f"success-shaped empty output indistinguishable from a genuine negative. "
            f"Remediation: {self.remediation}"
        )


def require_input(
    *,
    stage: str,
    present: bool,
    input_desc: str,
    produced_by: str,
    remediation: str,
) -> None:
    """Raise :class:`MissingInputError` when a required input is absent.

    ``present`` is the caller's own presence check — ``path.exists()`` for a
    single artifact, ``bool(list(dir.glob(pattern)))`` for a globbed set. Passing
    the boolean (rather than a path) keeps this one helper serving both the
    single-file and glob sites, and — critically — lets the caller draw the line
    between ABSENT (raise here) and PRESENT-BUT-EMPTY (the caller's own honest
    empty return), which is the distinction da#361 exists to preserve.
    """
    if not present:
        raise MissingInputError(
            stage=stage,
            input_desc=input_desc,
            produced_by=produced_by,
            remediation=remediation,
        )
