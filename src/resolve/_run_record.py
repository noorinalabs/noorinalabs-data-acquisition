"""The resolve run's own record of what it wrote — ``curated/_resolve_run.txt`` (da#428).

``prune-narrators`` DETACH-DELETEs every ``Narrator`` absent from the canonical
keep-set. da#426 measured a *valid but truncated* ``narrators_canonical.parquet``
deleting **83.9% of the graph** with ``missing == 0`` and every derived gate green:
every gate in the prune path is derived from the keep-set it is meant to validate,
so all of them prove the prune obeyed the parquet and none can ask whether it was
the *right* parquet — nor whether it was the *whole* parquet.

The missing signal is **completeness**, and it cannot be recovered downstream:

* The parquet's own footer ``num_rows`` is self-consistent on a short file.
* B2's per-object md5 binds the file to itself; a truncated-but-intact upload has
  a perfectly good hash.
* ``publish_parquet.py`` is a **separate process** that starts after resolve has
  exited and reads ``data/curated/`` off disk. It holds no tally, so any count it
  computed could only be a READ-BACK of the parquet it is about to upload — which
  agrees with a short file by construction.

**The tally exists only while the writing process is alive.** So the process that
*wrote* the parquet must declare, before the write, how many canonical ids it
intended to write; that declaration is this module's record, and it is the only
number in the whole prune path not derived from the artifact being validated.

The consumer contract (noorinalabs-deploy ``scripts/verify_prune_provenance.sh``)::

    RESOLVE_RUN canonical_ids=<n> run_status=complete git_sha=<sha>

Two artifacts, each declaring only what its author can honestly attest: this one
carries the count and the run's terminal status; ``publish_parquet.py``'s
``_manifest.txt`` carries md5 + bytes and is FORBIDDEN from carrying a count. The
gate refuses a manifest that declares ``canonical_ids`` at all — a publisher that
thinks it knows the count has read the file back.

Why the tally is bound to the WRITE and not to a named stage
------------------------------------------------------------
``narrators_canonical.parquet`` does not have "a producing stage". It has **six**
writers, every one of them in :data:`~src.resolve.RESOLVE_STEP_ORDER`:

===============  ==========================================================
stage            effect on ``narrators_canonical.parquet``
===============  ==========================================================
``disambiguate`` OVERWRITES it
``bio_promote``  MERGEs into it (43,109 of 129,234 ids are bio-only, da#352)
``cluster``      fuzzy-clusters in place, DROPPING merged ids
``narrator_split`` rewrites it with peeled rows APPENDED
``reconcile``    rewrites it (date fold; row set preserved, BYTES change)
``tabaqa_dates`` rewrites it (date fallback; row set preserved, BYTES change)
===============  ==========================================================

A tally taken at ``disambiguate`` declares ~86k while the file holds ~129k, so the
consumer's equality would **false-FAIL on every healthy run** — and a false-FAIL on
a destructive op sends an operator to "restore from backup" after a prune that did
exactly the right thing, after which the next person weakens the gate.

Naming the "last" writer is equally wrong: it is a fact about the *current* stage
list, and the last two writers were both added after the two this issue was filed
against. So the tally is not a property of any stage — it is a property of **the
write**. :func:`write_canonical` is the single sanctioned writer of the artifact,
it re-mints the tally on every call, and whichever writer happens to run last is
therefore the one whose tally survives. A seventh writer inherits the guarantee by
construction (and ``tests/test_resolve/test_run_record.py`` fails if one is added
that bypasses this function).

Why the record is fail-closed
-----------------------------
:func:`write_canonical` always stamps ``run_status=incomplete``: at the moment of
any write, the run has by definition not finished. Only :func:`finalize_run`, at
the successful end of ``run_all``, upgrades it to ``complete``. A process killed
mid-run, a stage that raised, a ``--stop-after`` probe — all leave ``incomplete``
on disk and the publish refuses. Nothing has to remember to mark failure.

``run_status`` describes **the run that minted the tally**, not merely the last run
to execute. A later ``--from-step dedup`` run touches no canonical writer, so it
leaves this record entirely alone rather than upgrading a partial run's tally to
``complete`` — which would be a false PASS reached by a legitimate operation
(da#268/da#272 resume).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from src.parse.base import write_parquet
from src.resolve.schemas import NARRATORS_CANONICAL_SCHEMA
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pyarrow as pa

logger = get_logger(__name__)

__all__ = [
    "CANONICAL_PARQUET_NAME",
    "RESOLVE_RUN_FILENAME",
    "ResolveRunRecord",
    "RunStatus",
    "canonical_id_tally",
    "current_run_id",
    "file_md5",
    "finalize_run",
    "read_run_record",
    "write_canonical",
]

CANONICAL_PARQUET_NAME = "narrators_canonical.parquet"

# The producer half of the da#428 contract. Read by noorinalabs-deploy's
# scripts/verify_prune_provenance.sh, which requires this object to exist and to
# carry BOTH canonical_ids and run_status before any node may be deleted.
RESOLVE_RUN_FILENAME = "_resolve_run.txt"

# The record keyword the consumer greps (`grep '^RESOLVE_RUN '`, first match wins).
# Per-stage forensics go on RESOLVE_STAGE lines so they can never shadow it.
_RUN_KEYWORD = "RESOLVE_RUN"
_STAGE_KEYWORD = "RESOLVE_STAGE"


class RunStatus(StrEnum):
    """Whether the run that minted the tally reached its terminal stage.

    Lowercase-and-underscore because the consumer extracts it with the charclass
    ``[a-z_]``; a value outside that alphabet reads back as EMPTY, and an empty
    read that nobody notices is precisely the silent zero this contract exists to
    refuse (``feedback_silent_zero_is_not_a_measurement``).

    The distinction is load-bearing and no count can supply it: a run that halts
    after writing a coherent partial file is internally consistent, and its tally —
    taken before that write — matches what it managed to write. Only the producer
    can attest that it reached the end, so it must, explicitly.
    """

    COMPLETE = "complete"
    """``run_all`` reached its terminal stage with no stage in error."""

    INCOMPLETE = "incomplete"
    """A write happened but the run has not (yet) finished. The default, always."""


@dataclass(frozen=True)
class ResolveRunRecord:
    """What the writing process declares about the canonical parquet it produced.

    ``canonical_ids`` is the in-memory tally taken BEFORE the write.
    ``parquet_md5`` binds that tally to the exact bytes it was taken over — see
    :func:`write_canonical` for why that binding is not a read-back.
    """

    canonical_ids: int
    run_status: RunStatus
    git_sha: str
    run_id: str
    last_writer: str
    parquet_md5: str
    parquet_bytes: int
    stages: tuple[str, ...] = ()

    def render(self) -> str:
        """Serialise to the ``key=value`` instrument-line format the consumer parses.

        Deliberately NOT JSON: the consumer parses this on the VPS inside an
        ``ssh-action`` script, where no ``jq`` and no ``python3`` is used. This is
        the same shape as ``PRUNE_NARRATORS_SUMMARY``, which both repos already
        parse with an anchored whole-token extraction.
        """
        head = " ".join(
            (
                _RUN_KEYWORD,
                f"canonical_ids={self.canonical_ids}",
                f"run_status={self.run_status.value}",
                f"git_sha={self.git_sha}",
                f"run_id={self.run_id}",
                f"last_writer={self.last_writer}",
                f"parquet_md5={self.parquet_md5}",
                f"parquet_bytes={self.parquet_bytes}",
            )
        )
        lines = [head, *(f"{_STAGE_KEYWORD} {s}" for s in self.stages)]
        return "\n".join(lines) + "\n"


def canonical_id_tally(table: pa.Table) -> int:
    """Count DISTINCT NON-EMPTY ``canonical_id`` values in an in-memory table.

    **The definition must match the consumer's exactly.** ``read_canonical_ids()``
    (:mod:`src.graph.prune`) returns a ``set`` and filters falsy ids::

        ids = {cid for cid in table.column("canonical_id").to_pylist() if cid}

    so the ``canonical=`` the prune reports is a distinct, non-empty count. A naive
    ``len(df)`` here would false-FAIL the equality on any duplicate or null id — on
    a *destructive* op, which is how a good gate gets weakened.

    Same definition, different derivation: this one reads the rows the writer is
    about to hand to parquet; the consumer reads the file that arrived. Both halves
    are load-bearing, and the whole binding is the gap between them.
    """
    return len({cid for cid in table.column("canonical_id").to_pylist() if cid})


def file_md5(path: Path) -> str:
    """md5 of ``path``'s bytes, streamed."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    """Short SHA of the code that produced this run, or ``unknown``.

    Provenance, not a gate: it makes a hand-authored record an act of forgery
    rather than an act of convenience. Never raises — a publish must not fail
    because the producer was run from a tarball.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


_RUN_ID = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"


def current_run_id() -> str:
    """This process's run id — stable for the life of the process.

    Process-scoped rather than threaded through every stage signature, because the
    resolve stages all execute in one process. It is what lets :func:`finalize_run`
    ask "did *I* mint this tally?" without enumerating writer stages by name.
    """
    return _RUN_ID


def write_canonical(table: pa.Table, canonical_path: Path, *, stage: str) -> Path:
    """Write ``narrators_canonical.parquet`` AND declare the tally. The only writer.

    The tally is computed from ``table`` — the in-memory rows — **before**
    :func:`write_parquet` is called. It is therefore not a function of the file, and
    a file that ends up short (a partial write, a truncation, a botched copy) does
    NOT agree with it. That disagreement is the entire completeness signal, and it
    is the property ``tests/test_resolve/test_run_record.py::test_tally_survives_
    truncation_of_the_written_parquet`` pins: truncate the parquet after the write
    and the record still declares the original count.

    ``parquet_md5`` is taken over the bytes just written. That is a hash of the
    file, NOT a count of it — it cannot produce ``canonical_ids`` and does not
    participate in deriving it, so it introduces no circularity. Its one job is to
    bind the in-memory tally to the exact bytes it describes, so that a parquet
    later rewritten by anything that did NOT come through this function (a scrub
    script, a hand edit, a stale file left by a resumed run) fails to match and the
    publish refuses. The count says *how many rows were meant*; the md5 says *which
    bytes were meant*. Neither substitutes for the other.

    ``run_status`` is stamped INCOMPLETE unconditionally: at the moment of a write,
    the run has not finished. Only :func:`finalize_run` upgrades it.
    """
    tally = canonical_id_tally(table)

    write_parquet(table, canonical_path, schema=NARRATORS_CANONICAL_SCHEMA)

    record = ResolveRunRecord(
        canonical_ids=tally,
        run_status=RunStatus.INCOMPLETE,
        git_sha=_git_sha(),
        run_id=current_run_id(),
        last_writer=stage,
        parquet_md5=file_md5(canonical_path),
        parquet_bytes=canonical_path.stat().st_size,
    )
    _write_record(canonical_path.parent, record)

    logger.info(
        "canonical_tally_declared",
        stage=stage,
        canonical_ids=tally,
        rows=table.num_rows,
        path=str(canonical_path),
    )
    return canonical_path


def _record_path(curated_dir: Path) -> Path:
    return curated_dir / RESOLVE_RUN_FILENAME


def _write_record(curated_dir: Path, record: ResolveRunRecord) -> Path:
    path = _record_path(curated_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.render(), encoding="utf-8")
    return path


def read_run_record(curated_dir: Path) -> ResolveRunRecord | None:
    """Parse ``curated/_resolve_run.txt``, or ``None`` if it is absent/unreadable.

    ``None`` means the keep-set carries no declaration from the process that wrote
    it — which the publish treats as a hard refusal, never as a zero.
    """
    path = _record_path(curated_dir)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    head = next((ln for ln in text.splitlines() if ln.startswith(f"{_RUN_KEYWORD} ")), None)
    if head is None:
        return None
    tokens = dict(
        tok.split("=", 1) for tok in head.split(" ")[1:] if "=" in tok and not tok.startswith("=")
    )
    try:
        status = RunStatus(tokens["run_status"])
        return ResolveRunRecord(
            canonical_ids=int(tokens["canonical_ids"]),
            run_status=status,
            git_sha=tokens.get("git_sha", "unknown"),
            run_id=tokens.get("run_id", ""),
            last_writer=tokens.get("last_writer", ""),
            parquet_md5=tokens.get("parquet_md5", ""),
            parquet_bytes=int(tokens.get("parquet_bytes", "0")),
            stages=tuple(
                ln.removeprefix(f"{_STAGE_KEYWORD} ")
                for ln in text.splitlines()
                if ln.startswith(f"{_STAGE_KEYWORD} ")
            ),
        )
    except (KeyError, ValueError):
        return None


def finalize_run(curated_dir: Path, stages: tuple[str, ...], *, complete: bool) -> None:
    """Stamp the run's terminal status onto the record, at the end of ``run_all``.

    Upgrades ``run_status`` to ``complete`` **only if this process minted the tally**
    (``record.run_id == current_run_id()``). A resumed run that started past every
    canonical writer (``--from-step dedup``) changed no canonical bytes, so it must
    not re-status a record it did not produce: doing so would let a *partial* earlier
    run's tally be promoted to ``complete`` by a later run that merely finished — a
    false PASS reached by an entirely legitimate operation.

    Absent record = a run that wrote no canonical parquet and found none. Nothing to
    finalize; the publish refuses on the absence, which is the correct refusal.
    """
    record = read_run_record(curated_dir)
    if record is None:
        return
    if record.run_id != current_run_id():
        logger.info(
            "resolve_run_record_not_ours",
            minted_by=record.run_id,
            this_run=current_run_id(),
            run_status=record.run_status.value,
        )
        return

    status = RunStatus.COMPLETE if complete else RunStatus.INCOMPLETE
    from dataclasses import replace

    _write_record(curated_dir, replace(record, run_status=status, stages=stages))
    logger.info(
        "resolve_run_record_finalized",
        run_status=status.value,
        canonical_ids=record.canonical_ids,
        last_writer=record.last_writer,
    )
