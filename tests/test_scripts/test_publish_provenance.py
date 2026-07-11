"""The publisher half of the prune completeness binding (da#428).

``publish_parquet.py`` is a SEPARATE PROCESS that starts after resolve has exited. It
holds no tally, so any count it emitted could only be a read-back of the parquet it is
about to upload — which agrees perfectly with a truncated one. These tests pin the two
properties that make that impossible rather than merely discouraged:

1. it REFUSES to publish a keep-set whose tally is absent, partial, or describes
   different bytes than the ones about to be shipped; and
2. it is STRUCTURALLY incapable of counting — it never imports pyarrow.

Each refusal asserts the REASON, not merely a non-zero exit: adding a guard earlier can
silently un-test every guard behind it, so "it raised" is not evidence that the door
under test is the one that fired.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.publish_parquet import (
    CANONICAL_PARQUET_NAME,
    CURATED_REQUIRED,
    RESOLVE_RUN_FILENAME,
    PlannedUpload,
    PublishError,
    assert_run_publishable,
    file_md5,
    manifest_line,
    parse_resolve_run,
    render_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _curated(tmp_path: Path, *, status: str = "complete", canonical_ids: int = 129234) -> Path:
    curated = tmp_path / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    for name in CURATED_REQUIRED:
        (curated / name).write_bytes(b"PAR1" + name.encode())
    md5 = file_md5(curated / CANONICAL_PARQUET_NAME)
    (curated / RESOLVE_RUN_FILENAME).write_text(
        f"RESOLVE_RUN canonical_ids={canonical_ids} run_status={status} git_sha=abc1234 "
        f"run_id=1-dead last_writer=tabaqa_dates parquet_md5={md5} parquet_bytes=8\n"
        "RESOLVE_STAGE step=ner outcome=ran files=1\n"
    )
    return curated


def _decl(curated: Path):  # noqa: ANN202 — test helper
    return parse_resolve_run((curated / RESOLVE_RUN_FILENAME).read_text())


# --- The record parses exactly as the consumer parses it ---------------------
def test_parses_the_consumer_contract_line(tmp_path: Path) -> None:
    curated = _curated(tmp_path)
    decl = _decl(curated)
    assert decl is not None
    assert decl.canonical_ids == 129234
    assert decl.run_status == "complete"
    assert decl.parquet_md5 == file_md5(curated / CANONICAL_PARQUET_NAME)


def test_whole_token_extraction_cannot_be_hijacked_by_a_longer_key() -> None:
    """``parquet_md5=`` must never answer for ``md5=`` (the da#419 whole-token rule)."""
    decl = parse_resolve_run(
        "RESOLVE_RUN canonical_ids=7 run_status=complete parquet_md5=deadbeef "
        "extra_canonical_ids=9\n"
    )
    assert decl is not None
    assert decl.canonical_ids == 7  # NOT 9
    assert decl.parquet_md5 == "deadbeef"


# --- The refusal doors. Each asserts its own REASON. -------------------------
def test_refuses_when_the_record_is_absent(tmp_path: Path) -> None:
    curated = _curated(tmp_path)
    (curated / RESOLVE_RUN_FILENAME).unlink()

    with pytest.raises(PublishError, match="absent or malformed"):
        assert_run_publishable(None, file_md5(curated / CANONICAL_PARQUET_NAME))


def test_refuses_when_the_run_did_not_complete(tmp_path: Path) -> None:
    """The 'resolve stopped early' mode. No count equality of any kind can see it: the
    tally and the file are both short and perfectly consistent."""
    curated = _curated(tmp_path, status="incomplete")

    with pytest.raises(PublishError, match="did NOT complete"):
        assert_run_publishable(_decl(curated), file_md5(curated / CANONICAL_PARQUET_NAME))


def test_refuses_a_zero_id_keep_set(tmp_path: Path) -> None:
    curated = _curated(tmp_path, canonical_ids=0)

    with pytest.raises(PublishError, match="delete every narrator"):
        assert_run_publishable(_decl(curated), file_md5(curated / CANONICAL_PARQUET_NAME))


def test_refuses_when_the_tally_describes_different_bytes(tmp_path: Path) -> None:
    """The skipped-tally-stage door, and the scrub-script door, are the same door.

    ``--from-step`` (da#268/da#272) can start past every canonical writer, so no tally is
    minted this run; a scrub script can rewrite the parquet outside the sanctioned writer.
    Both leave a record describing bytes that are no longer on disk. A skipped tally is
    not a smaller number — it is NO number — and the publisher must never fall back to
    counting the file itself.
    """
    curated = _curated(tmp_path)

    # The mutation is EFFECTIVE: the parquet really changed after the tally was taken.
    before = file_md5(curated / CANONICAL_PARQUET_NAME)
    (curated / CANONICAL_PARQUET_NAME).write_bytes(b"PAR1-rewritten-by-a-scrub-script")
    after = file_md5(curated / CANONICAL_PARQUET_NAME)
    assert before != after, "mutation disarmed — the file did not actually change"

    with pytest.raises(PublishError, match="NOT the file resolve took its tally over"):
        assert_run_publishable(_decl(curated), after)


def test_accepts_a_complete_run_whose_tally_matches(tmp_path: Path) -> None:
    """The known-negative. Without this, every refusal above could be a guard that
    refuses everything — including the honest artifact the gate exists to let through."""
    curated = _curated(tmp_path)

    assert_run_publishable(_decl(curated), file_md5(curated / CANONICAL_PARQUET_NAME))  # no raise


# --- The manifest declares md5 + bytes, and NEVER a count --------------------
def test_manifest_carries_no_count_at_all(tmp_path: Path) -> None:
    """The consumer REFUSES a manifest that declares ``canonical_ids`` — a publisher that
    thinks it knows the count has read the file back. This asserts the exact grep the
    consumer runs (``^CANONICAL_MANIFEST .*canonical_ids=``) finds nothing."""
    curated = _curated(tmp_path)
    uploads = [PlannedUpload(curated / n, f"curated/{n}") for n in CURATED_REQUIRED]

    body = render_manifest(uploads)

    assert "canonical_ids" not in body
    assert "rows=" not in body
    for line in body.splitlines():
        assert line.startswith("CANONICAL_MANIFEST ")
        assert "canonical_ids=" not in line


def test_manifest_line_binds_each_curated_object_to_its_bytes(tmp_path: Path) -> None:
    curated = _curated(tmp_path)
    uploads = [PlannedUpload(curated / n, f"curated/{n}") for n in CURATED_REQUIRED]

    lines = render_manifest(uploads).strip().splitlines()

    assert len(lines) == len(CURATED_REQUIRED)
    canonical = next(ln for ln in lines if f"file={CANONICAL_PARQUET_NAME} " in ln + " ")
    assert f"md5={file_md5(curated / CANONICAL_PARQUET_NAME)}" in canonical
    assert f"bytes={(curated / CANONICAL_PARQUET_NAME).stat().st_size}" in canonical


def test_manifest_excludes_staging_objects(tmp_path: Path) -> None:
    curated = _curated(tmp_path)
    uploads = [
        PlannedUpload(curated / CANONICAL_PARQUET_NAME, f"curated/{CANONICAL_PARQUET_NAME}"),
        PlannedUpload(curated / CANONICAL_PARQUET_NAME, "staging/hadiths_bukhari.parquet"),
    ]
    assert len(render_manifest(uploads).strip().splitlines()) == 1


def test_manifest_line_format_is_the_consumer_token_shape() -> None:
    line = manifest_line("narrators_canonical.parquet", "0" * 32, 4096)
    assert line == (
        f"CANONICAL_MANIFEST file=narrators_canonical.parquet md5={'0' * 32} bytes=4096"
    )
    # Whole-token parseable: every token is exactly one `key=value`, no spaces in values.
    assert all(tok.count("=") == 1 for tok in line.split(" ")[1:])


# --- The structural property: the publisher CANNOT count ---------------------
def _imports_pyarrow(module: str) -> bool:
    """Import ``module`` in a FRESH interpreter and report whether pyarrow got pulled in.

    A subprocess is mandatory: inside the pytest process pyarrow is already in
    ``sys.modules`` (the producer tests import it), so an in-process check would report
    True for everything and be a detector that cannot separate its two classes.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}, sys; print('pyarrow' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return proc.stdout.strip() == "True"


def test_the_pyarrow_detector_separates_its_two_classes() -> None:
    """Verify the instrument before reading it: a module that DOES import pyarrow must
    come back True, or the assertion below proves nothing (a silent False for every
    input would 'pass' while measuring nothing)."""
    assert _imports_pyarrow("src.resolve._run_record") is True, (
        "detector is blind — it cannot see pyarrow even when present"
    )
    assert _imports_pyarrow("json") is False


def test_publish_parquet_cannot_count_because_it_never_imports_pyarrow() -> None:
    """The read-back is STRUCTURALLY impossible, not merely discouraged.

    If this module ever imports pyarrow — directly, or transitively by importing
    ``src.resolve`` — it acquires the ability to read the parquet back and declare a
    count, and the circularity is silently relocated from the operator to the publisher,
    wearing the words "producer-signed provenance" and a green md5. That is the exact
    failure da#428 exists to prevent, so it is asserted at the import graph, not in prose.
    """
    assert _imports_pyarrow("scripts.publish_parquet") is False, (
        "publish_parquet imports pyarrow — it can now read the parquet back and count it. "
        "The publisher must copy resolve's tally forward, never recompute it."
    )
