#!/usr/bin/env python3
"""Publish resolved Parquet to the ``noorinalabs-pipeline`` B2 bucket.

This is the reproducible **producer** half of the batch graph-load pipeline: it
takes the local resolve output (``data/curated/`` + ``data/staging/``) and copies
it to Backblaze B2 under a versioned, bucket-relative ``parquet_ref``, laid out in
the exact object layout the ``deploy-data-load.yml`` workflow (noorinalabs-deploy
#546 / PR #547) consumes. It retires the ad-hoc ``rclone copy`` box step so every
resolve run publishes the same layout the same way (owner directive: IaC over
box one-offs, da#342).

CONTRACT — objects published under ``noorinalabs-pipeline/<parquet_ref>/``:

    curated/_resolve_run.txt              (written by RESOLVE; copied forward, da#428)
    curated/_manifest.txt                 (written HERE, uploaded LAST; md5 + bytes ONLY)
    curated/narrators_canonical.parquet
    curated/narrator_mentions_resolved.parquet
    curated/narrator_mentions_resolved_muhaddithat.parquet
    staging/hadiths_*.parquet
    staging/collections_*.parquet
    staging/narrator_mentions_*.parquet   (full load only)
    staging/network_edges_*.parquet       (full load only)
    staging/parallel_links.parquet        (full load only)

``parquet_ref`` is a bucket-relative prefix (no bucket name, no leading slash),
defaulting to ``staged/narrator-resolve/<UTC-date>-<git-short-sha>`` so a load is
reproducible and a prior-good set is always re-loadable.

Credentials: rclone **native env config** only (``RCLONE_CONFIG_PIPELINE_*``,
sourced from the ``PIPELINE_B2_KEY_ID`` / ``PIPELINE_B2_KEY`` env secrets) — the
same credential-safe pattern ``scripts/backup.sh`` and ``deploy-data-load.yml``
use. No secret value ever lands on argv or in a log line (CWE-214). Uploads are
copy-only (``rclone copyto``) — never a ``sync --delete`` against the bucket.

Usage:
    make publish-parquet                       # full publish, default ref
    make publish-parquet DRY_RUN=true          # list, upload nothing
    make publish-parquet NODES_ONLY=true       # nodes-load set only
    PARQUET_REF=staged/narrator-resolve/... make publish-parquet
    python scripts/publish_parquet.py --parquet-ref <ref> [--nodes-only] [--dry-run]

The final ``parquet_ref`` is printed as the LAST line of stdout so the operator can
paste it straight into the ``deploy-data-load.yml`` dispatch. All diagnostics go to
stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import get_settings

# --- Contract constants -----------------------------------------------------
BUCKET = "noorinalabs-pipeline"
RCLONE_REMOTE = "PIPELINE"

# The three curated resolve outputs — required for BOTH nodes-only and full loads
# (Narrator nodes + the canonical_narrator_id-bearing resolved mentions the chain
# loader reads). A missing curated file is always a hard failure.
CURATED_REQUIRED: tuple[str, ...] = (
    "narrators_canonical.parquet",
    "narrator_mentions_resolved.parquet",
    "narrator_mentions_resolved_muhaddithat.parquet",
)
# Node-bearing staging Parquet — needed by both load modes.
STAGING_ALWAYS_GLOBS: tuple[str, ...] = (
    "hadiths_*.parquet",
    "collections_*.parquet",
)
# Edge-bearing staging Parquet — only published/required for a full nodes+edges
# load; a --nodes-only publish omits these (matching the consumer's mode gate).
STAGING_FULL_ONLY_GLOBS: tuple[str, ...] = (
    "narrator_mentions_*.parquet",
    "network_edges_*.parquet",
    "parallel_links.parquet",
)


CANONICAL_PARQUET_NAME = "narrators_canonical.parquet"

# --- The da#428 provenance contract -----------------------------------------
# Two artifacts, each declaring ONLY what its author can honestly attest.
#
#   curated/_resolve_run.txt   Written by RESOLVE (src/resolve/_run_record.py) — the
#                              process that WROTE the parquet. It alone holds an
#                              in-memory tally taken BEFORE the write, so its count
#                              does NOT agree with a short file, and it alone knows
#                              whether the run reached its terminal stage.
#
#                                RESOLVE_RUN canonical_ids=<n> run_status=complete git_sha=<sha>
#
#   curated/_manifest.txt      Written HERE. md5 + bytes ONLY — and it is FORBIDDEN
#                              from declaring a count.
#
#                                CANONICAL_MANIFEST file=<name> md5=<32hex> bytes=<n>
#
# WHY THIS SCRIPT MUST NOT DECLARE A COUNT, AND WHY IT CANNOT IMPORT THE PRODUCER
#
# This is a SEPARATE PROCESS. It starts after resolve has exited, reads data/curated/
# off disk and uploads it. Resolve's in-memory tally does not exist any more. So the
# only way this script could produce `canonical_ids` is by READING BACK the parquet it
# is about to upload — and a read-back agrees perfectly with a truncated file. It would
# be a gate certifying coverage it cannot see, wearing the words "producer-signed
# provenance" and a green md5. da#426 measured that shape deleting 83.9% of the graph.
#
# That is why this module imports NO pyarrow and does NOT import src.resolve (which
# would pull pyarrow in transitively and hand it the ability to count). The inability
# is STRUCTURAL, not a convention — `tests/test_scripts/test_publish_provenance.py`
# executes this module in a subprocess and asserts pyarrow never enters sys.modules.
# The consumer enforces the other side: noorinalabs-deploy's verify_prune_provenance.sh
# REFUSES outright a manifest that carries `canonical_ids=`.
#
# So publish COPIES THE TALLY FORWARD, byte for byte, and never recomputes it.
RESOLVE_RUN_FILENAME = "_resolve_run.txt"
MANIFEST_FILENAME = "_manifest.txt"

# Both provenance objects are REQUIRED — the consumer refuses the prune without either.
PROVENANCE_REQUIRED: tuple[str, ...] = (RESOLVE_RUN_FILENAME, MANIFEST_FILENAME)

_RUN_KEYWORD = "RESOLVE_RUN"
_MANIFEST_KEYWORD = "CANONICAL_MANIFEST"


class PublishError(RuntimeError):
    """A publish precondition or post-publish verification failed."""


@dataclass(frozen=True)
class PlannedUpload:
    """One local Parquet file mapped to its bucket-relative destination key."""

    local: Path
    remote_subpath: str  # e.g. "curated/narrators_canonical.parquet"


@dataclass(frozen=True)
class ResolveRunDeclaration:
    """What resolve declared about the canonical parquet — parsed, never recomputed.

    ``line`` is kept verbatim so the record is copied forward exactly as the producer
    wrote it. This script re-derives none of these fields.
    """

    canonical_ids: int
    run_status: str
    parquet_md5: str
    line: str


def parse_resolve_run(text: str) -> ResolveRunDeclaration | None:
    """Parse the ``RESOLVE_RUN`` line of ``curated/_resolve_run.txt``.

    ``None`` for absent/malformed — which :func:`assert_run_publishable` turns into a
    hard refusal. A missing instrument reading is a stop, never a silent zero.

    Whole-token extraction (``split`` on spaces, exact ``key=``), matching the
    consumer's ``token_value``: no key can be hijacked by a longer one ending in its
    name (``parquet_md5=`` can never answer for ``md5=``).
    """
    head = next((ln for ln in text.splitlines() if ln.startswith(f"{_RUN_KEYWORD} ")), None)
    if head is None:
        return None
    tokens: dict[str, str] = {}
    for tok in head.split(" ")[1:]:
        key, sep, value = tok.partition("=")
        if sep and key:
            tokens[key] = value
    try:
        return ResolveRunDeclaration(
            canonical_ids=int(tokens["canonical_ids"]),
            run_status=tokens["run_status"],
            parquet_md5=tokens.get("parquet_md5", ""),
            line=head,
        )
    except (KeyError, ValueError):
        return None


def assert_run_publishable(decl: ResolveRunDeclaration | None, canonical_md5: str) -> None:
    """Refuse to publish a keep-set that carries no honest, matching tally.

    Each door names a distinct failure, and none of them can be answered by reading
    the parquet — which is the point.
    """
    if decl is None:
        raise PublishError(
            f"curated/{RESOLVE_RUN_FILENAME} is absent or malformed. The keep-set carries no "
            "declaration from the process that WROTE it, and this publisher holds no tally of "
            "its own — it is a separate process and any count it computed would be a read-back "
            "of the parquet it is uploading, which agrees with a truncated file. Re-run resolve "
            "so it declares the tally (da#428); do NOT hand-author this file."
        )
    if decl.run_status != "complete":
        raise PublishError(
            f"the resolve run that produced this keep-set did NOT complete "
            f"(run_status={decl.run_status!r}). Its canonical set is partial BY THE PRODUCER'S "
            "OWN ACCOUNT — every id in it is real, so 'missing' would be 0 and every derived "
            "gate would go green while the prune deleted the narrators the run never got to "
            "name. No count equality of any kind can see this; only the producer can attest it."
        )
    if decl.canonical_ids <= 0:
        raise PublishError(
            f"resolve declares canonical_ids={decl.canonical_ids} — a prune against a zero-id "
            "keep-set would delete every narrator."
        )
    if decl.parquet_md5 != canonical_md5:
        raise PublishError(
            f"curated/{CANONICAL_PARQUET_NAME} is NOT the file resolve took its tally over.\n"
            f"  tally was taken over md5: {decl.parquet_md5}\n"
            f"  file on disk has     md5: {canonical_md5}\n"
            "The declared count therefore describes different bytes than the ones about to be "
            "uploaded, so it cannot bind them. This is what a resume that skipped every "
            "canonical-writing stage looks like, and what a scrub script that rewrote the "
            "parquet outside the sanctioned writer looks like. A skipped tally is not a smaller "
            "number — it is NO number. Re-run resolve through a canonical writer; the publisher "
            "must never fall back to counting the file itself."
        )


def file_md5(path: Path) -> str:
    """md5 of ``path``'s bytes, streamed. Transfer integrity — NOT a count."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_line(name: str, md5: str, size_bytes: int) -> str:
    """One ``CANONICAL_MANIFEST`` line. md5 + bytes ONLY — never a count."""
    return f"{_MANIFEST_KEYWORD} file={name} md5={md5} bytes={size_bytes}"


def render_manifest(uploads: list[PlannedUpload]) -> str:
    """The ``_manifest.txt`` body: one line per curated object.

    Transfer integrity is the only thing a publisher is in a position to know, so it
    is the only thing this declares. It carries no ``canonical_ids`` and no ``rows``:
    the consumer REFUSES a manifest that declares a count, because a publisher that
    thinks it knows the count has read the file back.
    """
    return (
        "\n".join(
            manifest_line(u.local.name, file_md5(u.local), u.local.stat().st_size)
            for u in uploads
            if u.remote_subpath.startswith("curated/")
        )
        + "\n"
    )


# --- Pure logic (unit-tested; no I/O) ---------------------------------------
def utc_date() -> str:
    """Today's UTC date as ``YYYY-MM-DD`` (the date half of the default ref)."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def default_parquet_ref(date_utc: str, short_sha: str) -> str:
    """Build the versioned default ref ``staged/narrator-resolve/<date>-<sha>``."""
    return f"staged/narrator-resolve/{date_utc}-{short_sha}"


def validate_parquet_ref(ref: str) -> str:
    """Validate ``ref`` is a clean bucket-relative prefix; return it normalized.

    Mirrors the input validation ``deploy-data-load.yml`` enforces: non-empty, no
    leading slash, and no leading bucket name.
    """
    ref = ref.strip().rstrip("/")
    if not ref:
        raise PublishError("parquet_ref is empty")
    if ref.startswith("/"):
        raise PublishError(
            f"parquet_ref must NOT start with '/' (it is a bucket-relative prefix): {ref!r}"
        )
    if ref == BUCKET or ref.startswith(f"{BUCKET}/"):
        raise PublishError(
            "parquet_ref must NOT include the bucket name; pass only the prefix under "
            f"{BUCKET} (e.g. staged/narrator-resolve/<ver>): {ref!r}"
        )
    return ref


def plan_curated(curated_dir: Path) -> list[PlannedUpload]:
    """Map the three required curated files; raise if any is missing locally."""
    planned: list[PlannedUpload] = []
    missing: list[str] = []
    for name in CURATED_REQUIRED:
        path = curated_dir / name
        if path.is_file():
            planned.append(PlannedUpload(path, f"curated/{name}"))
        else:
            missing.append(name)
    if missing:
        raise PublishError(
            f"missing required curated Parquet in {curated_dir}: {', '.join(missing)}"
        )
    return planned


def plan_staging(staging_dir: Path, nodes_only: bool) -> list[PlannedUpload]:
    """Map the staging Parquet for the requested load mode.

    ``hadiths_*`` and ``collections_*`` are required in both modes; the edge-bearing
    globs are required (and published) only for a full load. A required glob that
    matches nothing is a hard failure.
    """
    required = STAGING_ALWAYS_GLOBS + (() if nodes_only else STAGING_FULL_ONLY_GLOBS)
    planned: list[PlannedUpload] = []
    missing_globs: list[str] = []
    for pattern in required:
        matches = sorted(staging_dir.glob(pattern))
        if not matches:
            missing_globs.append(pattern)
            continue
        planned.extend(PlannedUpload(m, f"staging/{m.name}") for m in matches)
    if missing_globs:
        raise PublishError(
            f"no staging Parquet matched required pattern(s) in {staging_dir}: "
            f"{', '.join(missing_globs)}"
        )
    return planned


def build_upload_plan(
    curated_dir: Path, staging_dir: Path, nodes_only: bool
) -> list[PlannedUpload]:
    """Full local→remote object plan for the requested load mode."""
    return plan_curated(curated_dir) + plan_staging(staging_dir, nodes_only)


def verify_remote(listing: set[str], planned: list[PlannedUpload]) -> None:
    """Assert every planned object — and each required curated file — is in ``listing``.

    ``listing`` is the set of bucket-relative keys under ``<parquet_ref>/`` as
    returned by ``rclone lsf --recursive`` (e.g. ``curated/narrators_canonical.parquet``).
    Raises :class:`PublishError` loudly on any absence.

    The two provenance objects are required exactly as the curated parquets are
    (da#428). A ref whose parquets landed but whose ``_resolve_run.txt`` did not is a
    ref the consumer will refuse to prune against — and it must fail HERE, loudly, at
    publish time, rather than surfacing as a mysterious refusal on the production box
    at the moment someone is trying to run a destructive op.
    """
    uploaded_missing = [pu.remote_subpath for pu in planned if pu.remote_subpath not in listing]
    if uploaded_missing:
        raise PublishError(
            "post-publish verification failed — object(s) absent from B2 listing: "
            f"{', '.join(sorted(uploaded_missing))}"
        )
    required = [f"curated/{name}" for name in (*CURATED_REQUIRED, *PROVENANCE_REQUIRED)]
    curated_missing = [key for key in required if key not in listing]
    if curated_missing:
        raise PublishError(
            "post-publish verification failed — required curated object(s) missing: "
            f"{', '.join(curated_missing)}"
        )


# --- I/O boundary -----------------------------------------------------------
def remote_base(ref: str) -> str:
    """The rclone remote:path prefix for ``ref`` (bucket included)."""
    return f"{RCLONE_REMOTE}:{BUCKET}/{ref}"


def git_short_sha(repo_root: Path) -> str:
    """Short SHA of ``repo_root``'s HEAD, for the default versioned ref."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublishError(
            f"could not determine git short SHA for the default parquet_ref: {exc}"
        ) from exc
    sha = proc.stdout.strip()
    if not sha:
        raise PublishError("git rev-parse --short HEAD returned empty output")
    return sha


def build_rclone_env() -> dict[str, str]:
    """Return an env with rclone native-config vars for the pipeline B2 remote.

    Sources the account/key from the ``PIPELINE_B2_KEY_ID`` / ``PIPELINE_B2_KEY``
    env secrets. The values are only ever placed in the child-process environment —
    never on argv or in a log line (CWE-214).
    """
    key_id = os.environ.get("PIPELINE_B2_KEY_ID", "")
    key = os.environ.get("PIPELINE_B2_KEY", "")
    if not key_id or not key:
        raise PublishError(
            "PIPELINE_B2_KEY_ID and PIPELINE_B2_KEY must be set in the environment "
            "(rclone native-env credentials; never passed on argv)."
        )
    env = os.environ.copy()
    env["RCLONE_CONFIG_PIPELINE_TYPE"] = "b2"
    env["RCLONE_CONFIG_PIPELINE_ACCOUNT"] = key_id
    env["RCLONE_CONFIG_PIPELINE_KEY"] = key
    return env


def rclone_copy_file(
    upload: PlannedUpload, ref: str, env: dict[str, str], *, rclone_bin: str = "rclone"
) -> None:
    """Copy one file to its destination key (copy-only; never delete)."""
    dest = f"{remote_base(ref)}/{upload.remote_subpath}"
    log(f"  copy {upload.local} -> {dest}")
    subprocess.run([rclone_bin, "copyto", str(upload.local), dest], check=True, env=env)


def rclone_list(ref: str, env: dict[str, str], *, rclone_bin: str = "rclone") -> set[str]:
    """List bucket-relative keys under ``<ref>/`` via ``rclone lsf --recursive``."""
    proc = subprocess.run(
        [rclone_bin, "lsf", "--recursive", remote_base(ref)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def log(msg: str) -> None:
    """Emit a diagnostic line to stderr (stdout is reserved for the final ref)."""
    print(msg, file=sys.stderr, flush=True)


# --- CLI --------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish resolved Parquet to the noorinalabs-pipeline B2 bucket "
            "(producer for deploy-data-load.yml)."
        )
    )
    parser.add_argument(
        "--parquet-ref",
        default=None,
        help=(
            "Bucket-relative B2 prefix (no bucket name, no leading slash). Default: "
            "PARQUET_REF env, else staged/narrator-resolve/<UTC-date>-<git-short-sha>."
        ),
    )
    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help=(
            "Publish only the nodes-load staging set (hadiths_*, collections_*); the "
            "full-load-only edge Parquet is omitted and not required."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the objects that would be published; upload nothing.",
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=None,
        help="Override the curated dir (default: settings.data_curated_dir).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Override the staging dir (default: settings.data_staging_dir).",
    )
    parser.add_argument(
        "--rclone-bin",
        default="rclone",
        help="rclone binary to invoke (default: rclone).",
    )
    return parser.parse_args(argv)


def _run(argv: list[str] | None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    settings = get_settings()
    curated_dir: Path = args.curated_dir or settings.data_curated_dir
    staging_dir: Path = args.staging_dir or settings.data_staging_dir

    ref = (
        args.parquet_ref
        or os.environ.get("PARQUET_REF")
        or default_parquet_ref(utc_date(), git_short_sha(repo_root))
    )
    ref = validate_parquet_ref(ref)

    plan = build_upload_plan(curated_dir, staging_dir, args.nodes_only)

    mode = "nodes-only" if args.nodes_only else "full (nodes+edges)"
    log(f"[publish-parquet] parquet_ref = {ref}")
    log(f"[publish-parquet] mode        = {mode}")
    log(f"[publish-parquet] curated dir = {curated_dir}")
    log(f"[publish-parquet] staging dir = {staging_dir}")

    # --- The completeness gate (da#428) --------------------------------------
    # BEFORE anything is uploaded. Resolve's record is the only number in the whole
    # prune path not derived from the artifact being validated; if it is absent, if
    # the run did not finish, or if it describes different bytes than the ones about
    # to be shipped, we refuse — we never fall back to counting the parquet.
    canonical_local = curated_dir / CANONICAL_PARQUET_NAME
    canonical_md5 = file_md5(canonical_local)
    run_record = curated_dir / RESOLVE_RUN_FILENAME
    decl = (
        parse_resolve_run(run_record.read_text(encoding="utf-8")) if run_record.is_file() else None
    )
    assert_run_publishable(decl, canonical_md5)
    assert decl is not None  # narrowed by assert_run_publishable
    log(
        f"[publish-parquet] resolve declared canonical_ids={decl.canonical_ids} "
        f"run_status={decl.run_status} (tally minted in-memory by the writing process; "
        "this publisher computes no count)"
    )

    # Written from the same bytes the gate just approved. md5 + bytes ONLY.
    manifest_local = curated_dir / MANIFEST_FILENAME
    provenance = [
        PlannedUpload(run_record, f"curated/{RESOLVE_RUN_FILENAME}"),
        PlannedUpload(manifest_local, f"curated/{MANIFEST_FILENAME}"),
    ]
    total = len(plan) + len(provenance)
    log(f"[publish-parquet] {total} object(s) to publish under {BUCKET}/{ref}/")

    if args.dry_run:
        log("[publish-parquet] --dry-run: the following WOULD be published (nothing uploaded):")
        for upload in [*plan, *provenance]:
            log(f"  {upload.local} -> {remote_base(ref)}/{upload.remote_subpath}")
        print(ref)
        return 0

    if shutil.which(args.rclone_bin) is None:
        raise PublishError(f"rclone binary {args.rclone_bin!r} not found on PATH")

    manifest_local.write_text(render_manifest(plan), encoding="utf-8")

    env = build_rclone_env()
    for upload in plan:
        rclone_copy_file(upload, ref, env, rclone_bin=args.rclone_bin)

    # Provenance LAST, manifest last of all: its presence is itself the evidence that
    # the publish ran to completion. _resolve_run.txt is copied forward verbatim —
    # the publisher never re-derives what only the producer could honestly attest.
    for upload in provenance:
        rclone_copy_file(upload, ref, env, rclone_bin=args.rclone_bin)

    full_plan = [*plan, *provenance]
    log("[publish-parquet] verifying published objects (rclone lsf)...")
    listing = rclone_list(ref, env, rclone_bin=args.rclone_bin)
    verify_remote(listing, full_plan)
    log(f"[publish-parquet] OK - {len(full_plan)} object(s) verified present under {BUCKET}/{ref}/")

    # Final stdout line: the ref for the operator to paste into deploy-data-load.yml.
    print(ref)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except PublishError as exc:
        log(f"ERROR: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        log(f"ERROR: rclone command failed (exit {exc.returncode}): {' '.join(map(str, exc.cmd))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
