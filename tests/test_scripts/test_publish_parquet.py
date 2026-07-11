"""Tests for the da#342 publish-parquet helper (``scripts/publish_parquet.py``).

These exercise the pure logic (ref defaulting/validation, local→remote layout
mapping, post-publish presence verification) plus the CLI wiring with rclone
mocked — no real B2 access. The credential-safety invariant (no secret value on
argv) is asserted directly against the recorded subprocess calls.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.publish_parquet import (
    BUCKET,
    CANONICAL_PARQUET_NAME,
    CURATED_REQUIRED,
    MANIFEST_FILENAME,
    PROVENANCE_REQUIRED,
    RESOLVE_RUN_FILENAME,
    PlannedUpload,
    PublishError,
    build_rclone_env,
    default_parquet_ref,
    file_md5,
    main,
    plan_curated,
    plan_staging,
    validate_parquet_ref,
    verify_remote,
)

_FAKE_KEY_ID = "0011223344556677deadbeef"
_FAKE_KEY = "K0ffeeBabeSecretAppKeyValue999"


def _make_curated(curated_dir: Path) -> None:
    """The three curated parquets PLUS the resolve run record they cannot publish without.

    The record is what resolve writes (da#428): an in-memory tally, the run's terminal
    status, and the md5 of the bytes the tally was taken over. Publish refuses without it.
    """
    curated_dir.mkdir(parents=True, exist_ok=True)
    for name in CURATED_REQUIRED:
        (curated_dir / name).write_bytes(b"PAR1")
    _write_run_record(curated_dir)


def _write_run_record(
    curated_dir: Path, *, canonical_ids: int = 129234, status: str = "complete"
) -> None:
    md5 = file_md5(curated_dir / CANONICAL_PARQUET_NAME)
    (curated_dir / RESOLVE_RUN_FILENAME).write_text(
        f"RESOLVE_RUN canonical_ids={canonical_ids} run_status={status} git_sha=abc1234 "
        f"run_id=1-deadbeef last_writer=tabaqa_dates parquet_md5={md5} parquet_bytes=4\n"
    )


def _make_staging(staging_dir: Path, *, full: bool) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "hadiths_bukhari.parquet").write_bytes(b"PAR1")
    (staging_dir / "collections_all.parquet").write_bytes(b"PAR1")
    if full:
        (staging_dir / "narrator_mentions_all.parquet").write_bytes(b"PAR1")
        (staging_dir / "network_edges_studied.parquet").write_bytes(b"PAR1")
        (staging_dir / "parallel_links.parquet").write_bytes(b"PAR1")


# --- ref defaulting / validation --------------------------------------------
def test_default_parquet_ref_shape() -> None:
    assert default_parquet_ref("2026-07-08", "abc1234") == (
        "staged/narrator-resolve/2026-07-08-abc1234"
    )


def test_validate_parquet_ref_accepts_clean_prefix() -> None:
    assert validate_parquet_ref("staged/narrator-resolve/2026-07-08-abc1234") == (
        "staged/narrator-resolve/2026-07-08-abc1234"
    )


def test_validate_parquet_ref_strips_trailing_slash() -> None:
    assert validate_parquet_ref("staged/x/") == "staged/x"


@pytest.mark.parametrize("bad", ["", "   ", "/leading/slash", f"{BUCKET}/x", BUCKET])
def test_validate_parquet_ref_rejects_bad(bad: str) -> None:
    with pytest.raises(PublishError):
        validate_parquet_ref(bad)


# --- curated layout mapping -------------------------------------------------
def test_plan_curated_maps_all_three(tmp_path: Path) -> None:
    _make_curated(tmp_path)
    plan = plan_curated(tmp_path)
    assert {pu.remote_subpath for pu in plan} == {f"curated/{n}" for n in CURATED_REQUIRED}
    assert all(pu.local.is_file() for pu in plan)


def test_plan_curated_missing_file_raises(tmp_path: Path) -> None:
    _make_curated(tmp_path)
    (tmp_path / "narrators_canonical.parquet").unlink()
    with pytest.raises(PublishError, match="narrators_canonical.parquet"):
        plan_curated(tmp_path)


# --- staging layout mapping -------------------------------------------------
def test_plan_staging_full_includes_edge_globs(tmp_path: Path) -> None:
    _make_staging(tmp_path, full=True)
    subpaths = {pu.remote_subpath for pu in plan_staging(tmp_path, nodes_only=False)}
    assert subpaths == {
        "staging/hadiths_bukhari.parquet",
        "staging/collections_all.parquet",
        "staging/narrator_mentions_all.parquet",
        "staging/network_edges_studied.parquet",
        "staging/parallel_links.parquet",
    }


def test_plan_staging_nodes_only_excludes_edge_globs(tmp_path: Path) -> None:
    # Edge files present on disk, but a nodes-only publish must NOT include them.
    _make_staging(tmp_path, full=True)
    subpaths = {pu.remote_subpath for pu in plan_staging(tmp_path, nodes_only=True)}
    assert subpaths == {
        "staging/hadiths_bukhari.parquet",
        "staging/collections_all.parquet",
    }


def test_plan_staging_missing_always_glob_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "collections_all.parquet").write_bytes(b"PAR1")  # no hadiths_*
    with pytest.raises(PublishError, match="hadiths_"):
        plan_staging(tmp_path, nodes_only=True)


def test_plan_staging_full_missing_edge_glob_raises(tmp_path: Path) -> None:
    _make_staging(tmp_path, full=False)  # only the always-globs present
    with pytest.raises(PublishError, match="network_edges_"):
        plan_staging(tmp_path, nodes_only=False)


def test_plan_staging_nodes_only_tolerates_absent_edges(tmp_path: Path) -> None:
    _make_staging(tmp_path, full=False)
    # Should not raise — edge files legitimately absent for a nodes-only publish.
    plan_staging(tmp_path, nodes_only=True)


# --- post-publish verification ----------------------------------------------
def test_verify_remote_passes_when_all_present() -> None:
    plan = [
        PlannedUpload(Path("x"), f"curated/{CURATED_REQUIRED[0]}"),
        PlannedUpload(Path("y"), "staging/hadiths_bukhari.parquet"),
    ]
    listing = {
        *(f"curated/{n}" for n in CURATED_REQUIRED),
        *(f"curated/{n}" for n in PROVENANCE_REQUIRED),
        "staging/hadiths_bukhari.parquet",
    }
    verify_remote(listing, plan)  # no raise


def test_verify_remote_missing_planned_object_raises() -> None:
    plan = [PlannedUpload(Path("y"), "staging/hadiths_bukhari.parquet")]
    listing = {f"curated/{n}" for n in CURATED_REQUIRED}  # hadiths absent
    with pytest.raises(PublishError, match="absent from B2 listing"):
        verify_remote(listing, plan)


def test_verify_remote_missing_curated_raises() -> None:
    plan: list[PlannedUpload] = []
    listing = {f"curated/{CURATED_REQUIRED[0]}"}  # only 1 of 3 curated present
    with pytest.raises(PublishError, match="required curated object"):
        verify_remote(listing, plan)


@pytest.mark.parametrize("absent", PROVENANCE_REQUIRED)
def test_verify_remote_requires_both_provenance_objects(absent: str) -> None:
    """A ref whose parquets landed but whose provenance did not is a ref the consumer
    REFUSES to prune against. It must fail here, at publish, and not surface as a
    mysterious refusal on the production box mid-destructive-op (da#428)."""
    plan: list[PlannedUpload] = []
    listing = {
        *(f"curated/{n}" for n in CURATED_REQUIRED),
        *(f"curated/{n}" for n in PROVENANCE_REQUIRED if n != absent),
    }
    with pytest.raises(PublishError, match=f"curated/{absent}"):
        verify_remote(listing, plan)


# --- credential handling ----------------------------------------------------
def test_build_rclone_env_requires_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_B2_KEY_ID", raising=False)
    monkeypatch.delenv("PIPELINE_B2_KEY", raising=False)
    with pytest.raises(PublishError, match="PIPELINE_B2_KEY_ID"):
        build_rclone_env()


def test_build_rclone_env_sets_native_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_B2_KEY_ID", _FAKE_KEY_ID)
    monkeypatch.setenv("PIPELINE_B2_KEY", _FAKE_KEY)
    env = build_rclone_env()
    assert env["RCLONE_CONFIG_PIPELINE_TYPE"] == "b2"
    assert env["RCLONE_CONFIG_PIPELINE_ACCOUNT"] == _FAKE_KEY_ID
    assert env["RCLONE_CONFIG_PIPELINE_KEY"] == _FAKE_KEY


# --- The gate is wired into main(), not merely defined (da#429 review) --------
@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("incomplete", "did NOT complete"),
        ("md5", "NOT the file resolve took its tally over"),
        ("absent", "absent or malformed"),
    ],
)
def test_main_refuses_a_bad_keep_set_and_uploads_NOTHING(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    reason: str,
) -> None:
    """``main()`` must actually CALL ``assert_run_publishable`` — and upload nothing.

    Every other refusal test in this suite calls ``assert_run_publishable`` directly as a
    unit, which pins the FUNCTION but not the BRANCH THAT CALLS IT. Oyunbileg Batbayar
    deleted the call from ``_run()`` and all 38 publish tests still passed — while
    ``rclone copyto ... narrators_canonical.parquet`` actually fired and an unbacked
    keep-set reached the upload. That is the exact failure this PR names in
    ``test_no_resolve_module_writes_the_canonical_master_directly`` and then committed
    itself, one layer up.

    The reason is asserted, not merely a non-zero exit: adding a guard EARLIER can
    silently un-test every guard behind it, so "it refused" is not evidence that the door
    under test is the one that fired.
    """
    curated = tmp_path / "curated"
    _make_curated(curated)
    _make_staging(tmp_path / "staging", full=True)
    monkeypatch.setenv("PIPELINE_B2_KEY_ID", _FAKE_KEY_ID)
    monkeypatch.setenv("PIPELINE_B2_KEY", _FAKE_KEY)
    monkeypatch.setattr("scripts.publish_parquet.shutil.which", lambda _b: "/usr/bin/rclone")

    if defect == "incomplete":
        _write_run_record(curated, status="incomplete")
    elif defect == "md5":
        # The tally was taken over other bytes — a resume that skipped every canonical
        # writer, or a scrub script that rewrote the parquet outside the sanctioned writer.
        (curated / CANONICAL_PARQUET_NAME).write_bytes(b"PAR1-rewritten-after-the-tally")
    elif defect == "absent":
        (curated / RESOLVE_RUN_FILENAME).unlink()

    calls: list[list[str]] = []

    def _record(cmd: list[str], **_kwargs: object) -> object:
        calls.append(cmd)
        raise AssertionError(f"rclone must never be invoked for a refused keep-set: {cmd}")

    monkeypatch.setattr("scripts.publish_parquet.subprocess.run", _record)

    rc = main(
        [
            "--parquet-ref",
            "staged/narrator-resolve/2026-07-11-abc1234",
            "--curated-dir",
            str(curated),
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )

    assert rc != 0, "publish ACCEPTED an artifact the consumer will refuse"
    assert reason in capsys.readouterr().err, "refused — but NOT by the door under test"
    assert calls == [], "an unbacked keep-set was uploaded to B2"


def test_main_publishes_the_healthy_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The known-negative for the three refusals above.

    Without it, deleting the gate's call site and replacing it with an unconditional
    ``raise`` would satisfy every test above — a publisher that refuses EVERYTHING,
    including the healthy artifact it exists to ship.
    """
    curated = tmp_path / "curated"
    _make_curated(curated)
    _make_staging(tmp_path / "staging", full=True)
    monkeypatch.setenv("PIPELINE_B2_KEY_ID", _FAKE_KEY_ID)
    monkeypatch.setenv("PIPELINE_B2_KEY", _FAKE_KEY)
    monkeypatch.setattr("scripts.publish_parquet.shutil.which", lambda _b: "/usr/bin/rclone")

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "lsf" in cmd:
            listing = "\n".join(
                [
                    *(f"curated/{n}" for n in CURATED_REQUIRED),
                    *(f"curated/{n}" for n in PROVENANCE_REQUIRED),
                    "staging/hadiths_bukhari.parquet",
                    "staging/collections_all.parquet",
                    "staging/narrator_mentions_all.parquet",
                    "staging/network_edges_studied.parquet",
                    "staging/parallel_links.parquet",
                ]
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.publish_parquet.subprocess.run", _fake_run)

    rc = main(
        [
            "--parquet-ref",
            "staged/narrator-resolve/2026-07-11-abc1234",
            "--curated-dir",
            str(curated),
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )

    assert rc == 0, "the gate refused a HEALTHY artifact — a false-FAIL on the launch path"
    # The tally is copied forward verbatim; the publisher declares no count of its own.
    assert "canonical_ids=129234" in capsys.readouterr().err


# --- CLI wiring -------------------------------------------------------------
def test_dry_run_uploads_nothing_and_echoes_ref(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_curated(tmp_path / "curated")
    _make_staging(tmp_path / "staging", full=True)

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr("scripts.publish_parquet.subprocess.run", _boom)

    ref = "staged/narrator-resolve/2026-07-08-deadbee"
    rc = main(
        [
            "--dry-run",
            "--parquet-ref",
            ref,
            "--curated-dir",
            str(tmp_path / "curated"),
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == ref


def test_publish_copy_only_and_no_secret_on_argv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_curated(tmp_path / "curated")
    _make_staging(tmp_path / "staging", full=True)
    monkeypatch.setenv("PIPELINE_B2_KEY_ID", _FAKE_KEY_ID)
    monkeypatch.setenv("PIPELINE_B2_KEY", _FAKE_KEY)
    monkeypatch.setattr("scripts.publish_parquet.shutil.which", lambda _b: "/usr/bin/rclone")

    ref = "staged/narrator-resolve/2026-07-08-abc1234"
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "lsf" in cmd:
            # Echo back the exact contract layout as if B2 now holds it.
            listing = "\n".join(
                [
                    *(f"curated/{n}" for n in CURATED_REQUIRED),
                    *(f"curated/{n}" for n in PROVENANCE_REQUIRED),
                    "staging/hadiths_bukhari.parquet",
                    "staging/collections_all.parquet",
                    "staging/narrator_mentions_all.parquet",
                    "staging/network_edges_studied.parquet",
                    "staging/parallel_links.parquet",
                ]
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.publish_parquet.subprocess.run", _fake_run)

    rc = main(
        [
            "--parquet-ref",
            ref,
            "--curated-dir",
            str(tmp_path / "curated"),
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == ref

    # 8 data objects + the 2 provenance objects (da#428) copied, + 1 lsf verification.
    copy_calls = [c for c in calls if "copyto" in c]
    assert len(copy_calls) == 10

    # The manifest goes up LAST of all, so its mere presence in the bucket is itself the
    # evidence that the publish ran to completion — a half-finished publish cannot leave
    # a ref that looks complete to the consumer.
    assert copy_calls[-1][-1].endswith(f"/curated/{MANIFEST_FILENAME}")
    assert copy_calls[-2][-1].endswith(f"/curated/{RESOLVE_RUN_FILENAME}")

    # Copy-only: never a destructive rclone verb against the bucket.
    for cmd in calls:
        assert not ({"sync", "delete", "purge", "deletefile", "rmdir"} & set(cmd))
    # CWE-214: the secret VALUES never appear on any argv (they travel via env).
    flat = " ".join(tok for cmd in calls for tok in cmd)
    assert _FAKE_KEY not in flat
    assert _FAKE_KEY_ID not in flat


def test_publish_fails_loud_when_verification_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_curated(tmp_path / "curated")
    _make_staging(tmp_path / "staging", full=True)
    monkeypatch.setenv("PIPELINE_B2_KEY_ID", _FAKE_KEY_ID)
    monkeypatch.setenv("PIPELINE_B2_KEY", _FAKE_KEY)
    monkeypatch.setattr("scripts.publish_parquet.shutil.which", lambda _b: "/usr/bin/rclone")

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "lsf" in cmd:
            # A curated file failed to land — listing is short.
            return subprocess.CompletedProcess(
                cmd, 0, stdout="curated/narrators_canonical.parquet\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.publish_parquet.subprocess.run", _fake_run)

    rc = main(
        [
            "--parquet-ref",
            "staged/x/y",
            "--curated-dir",
            str(tmp_path / "curated"),
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )
    assert rc == 1  # verification failure surfaces as a non-zero exit
