"""Shared crash-resume checkpoint primitives for the resolve pipeline (da#272).

Unifies the seven conventions first established for ``disambiguate`` (da#268 /
PR #269) so every long-running resolve stage resumes after a host crash under
ONE shape instead of copy-pasting the pattern five times:

1. Checkpoint location: ``<staging_dir>/.<stage>_checkpoint/state.json``
   (:func:`checkpoint_dir`).
2. Atomic write: sibling ``.tmp`` then ``os.replace`` so a crash mid-write never
   leaves a half-file — the visible ``state.json`` is always complete or absent
   (:func:`save_checkpoint`).
3. Cadence: every N batches/blocks (stage-appropriate N), plus on clean stage
   exit (:func:`resolve_cadence` + the stage's loop).
4. Content fingerprint of the stage input — stable driver columns hashed in row
   order, never mtime — so a checkpoint taken against different input is
   discarded, never silently reused (:func:`hash_parquet_column_groups` /
   :func:`hash_strings`).
5. Cleared on stage success (:func:`clear_checkpoint`).
6. CLI ``resolve --from-step`` stays the step-level entry; intra-stage resume is
   automatic when a valid checkpoint exists; a uniform ``--no-resume`` forces a
   cold start (threaded to each stage's ``run(resume=...)``).
7. A single ``<stage>_resume`` log line on resume stating what was skipped
   (:func:`log_resume`).

Each stage owns the *shape* of its serialized state (the accumulators it must
restore) and the column set that fingerprints its input; this module owns the
mechanics. The persisted payload is JSON, so a stage must serialise any
set/tuple accumulator to a list before handing it to :func:`save_checkpoint` and
rebuild it on restore.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.utils.logging import get_logger

logger = get_logger(__name__)

_STATE_FILENAME = "state.json"

# ASCII unit / record separators keep the streamed field/row hashing
# unambiguous — a value that happens to contain the delimiter cannot forge a
# field or row boundary and collide two distinct inputs to one digest.
_FP_UNIT_SEP = "\x1f"
_FP_ROW_SEP = "\x1e"


def checkpoint_dir(staging_dir: Path, stage: str) -> Path:
    """Location of ``stage``'s crash-resume checkpoint (convention #1).

    ``<staging_dir>/.<stage>_checkpoint/``. The staging tree is gitignored, so
    the checkpoint never leaks into version control.
    """
    return staging_dir / f".{stage}_checkpoint"


def save_checkpoint(ckpt_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically persist ``payload`` as the checkpoint state (convention #2).

    Writes to a sibling ``.tmp`` then ``os.replace`` so a crash mid-write never
    leaves a half-written state file — the visible ``state.json`` is always a
    complete checkpoint (or absent, which reads as a cold start).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f"{_STATE_FILENAME}.tmp"
    final = ckpt_dir / _STATE_FILENAME
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, final)


def load_checkpoint(ckpt_dir: Path) -> dict[str, Any] | None:
    """Load the checkpoint payload, or ``None`` when absent/unreadable.

    An unreadable (torn/corrupt) state file is treated as absent — the stage
    cold-starts rather than crashing on a malformed resume.
    """
    final = ckpt_dir / _STATE_FILENAME
    if not final.exists():
        return None
    try:
        loaded = json.loads(final.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        logger.warning("checkpoint_unreadable", path=str(final))
        return None
    return loaded if isinstance(loaded, dict) else None


def clear_checkpoint(ckpt_dir: Path) -> None:
    """Remove the checkpoint dir after a successful complete run (convention #5).

    Best effort: a leftover checkpoint is fingerprint-guarded on the next run
    (convention #4), so a failed delete can never cause an incorrect resume — it
    only wastes the fingerprint recomputation that then discards it.
    """
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def resolve_cadence(override: int | None, env_var: str, default: int) -> int:
    """Effective checkpoint cadence (kwarg > env var > default), floored at 1.

    The convention #3 knob. A non-integer env value warns and falls back to the
    default rather than crashing a multi-hour stage over an ops typo.
    """
    if override is not None:
        return max(1, int(override))
    env = os.environ.get(env_var)
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            logger.warning("checkpoint_cadence_invalid", env_var=env_var, value=env)
    return max(1, default)


def hash_parquet_column_groups(
    paths: Path | list[Path],
    groups: dict[str, tuple[str, ...]],
) -> tuple[dict[str, str], int]:
    """Content fingerprint (convention #4) of one or more parquet inputs.

    Streams every path row-group by row-group and, in that single pass, updates
    one SHA-256 per named *group* of columns over all rows in file+row order.
    Peak memory is one row-group, never the whole file. Order- (not mtime-)
    based: two byte-identical inputs fingerprint identically no matter when they
    were written, and any change to a hashed column's values flips exactly the
    digests of the groups that cover it — which is what lets a stage tell a
    genuinely changed input (discard the checkpoint) apart from an incidental
    rewrite of a column it does not depend on.

    ``paths`` may be a single path or a list hashed in the given order (a stage
    whose input is a glob of shards passes them pre-sorted for determinism).
    Returns ``(digest_by_group, total_rows)``; a missing/empty input yields
    empty-string digests and 0 rows so the caller cold-starts.

    The per-group digest is identical to hashing that group's columns alone:
    each row contributes ``UNIT_SEP.join(str(v) for the group's cols)`` followed
    by ``ROW_SEP`` to its own group hasher, independent of the other groups or of
    the physical column read order.
    """
    path_list = [paths] if isinstance(paths, Path) else list(paths)
    hashers = {name: hashlib.sha256() for name in groups}
    # Read the union of every group's columns once per row-group; index by name
    # so the (sorted) physical read order never perturbs a group's digest.
    all_cols = sorted({col for cols in groups.values() for col in cols})
    total = 0
    for path in path_list:
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=all_cols)
            n = table.num_rows
            total += n
            columns = {name: table.column(name).to_pylist() for name in all_cols}
            for i in range(n):
                for name, cols in groups.items():
                    row = _FP_UNIT_SEP.join(str(columns[c][i]) for c in cols)
                    hashers[name].update(row.encode("utf-8"))
                    hashers[name].update(_FP_ROW_SEP.encode("utf-8"))
    if total == 0:
        return {name: "" for name in groups}, 0
    return {name: h.hexdigest() for name, h in hashers.items()}, total


def hash_strings(*parts: object) -> str:
    """Content fingerprint for a stage input that is not a set of parquet columns.

    For stages (dedup, ner) whose resume-identity is a handful of scalars — an
    id-set hash already computed over the corpus, plus the params that change the
    output — join them into one SHA-256 so the same content-fingerprint
    discipline (convention #4) applies uniformly. Never incorporates mtime.
    """
    joined = _FP_ROW_SEP.join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def log_resume(stage: str, *, skipped: int, total: int, **extra: object) -> None:
    """Emit the uniform ``<stage>_resume`` progress line (convention #7)."""
    logger.info(
        f"{stage}_resume",
        checkpoint_skipped=skipped,
        total=total,
        pct=round(skipped / max(total, 1) * 100, 1),
        **extra,
    )
