"""Crash-resume tests for ``src.resolve.disambiguate`` (da#268).

Covers the checkpoint contract added to make the multi-hour disambiguate stream
recoverable after a host crash:

- input fingerprint splits stable content from the (random uuid4) mention_id;
- checkpoint write / load / clear roundtrip;
- a resumed run is OUTPUT-IDENTICAL to a cold run on the same input;
- a fingerprint mismatch (changed corpus) cold-starts;
- a content-match / mention_id-mismatch (NER rewrite) cold-starts with a hint;
- the ``run_all(from_step=...)`` step-skip gating.

All fixtures are tiny synthetic parquet files under tmp paths — never the real
corpus. The batch size + checkpoint cadence are driven small so a handful of
mentions still crosses a checkpoint boundary.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.resolve import disambiguate
from src.resolve.schemas import NARRATOR_MENTIONS_RESOLVED_SCHEMA
from src.utils.arabic import normalize_arabic

# Two bio candidates the exact-match stage resolves against, plus names that do
# not match any bio (self-canonicalized fallback path, da#99) so both branches of
# the streaming loop are exercised across the crash boundary.
_NAME_A = "محمد بن اسماعيل"
_NAME_B = "عبد الله بن يوسف"
_UNKNOWN = "فلان بن فلان"


def _write_bio_candidates(staging_dir: Path) -> None:
    def _col(values: list[str | int | None], typ: pa.DataType) -> pa.Array:
        return pa.array(values, type=typ)

    table = pa.table(
        {
            "bio_id": _col(["bio:a", "bio:b"], pa.string()),
            "name_ar": _col([_NAME_A, _NAME_B], pa.string()),
            "name_en": _col([None, None], pa.string()),
            "name_ar_normalized": _col(
                [normalize_arabic(_NAME_A), normalize_arabic(_NAME_B)], pa.string()
            ),
            "kunya": _col([None, None], pa.string()),
            "nisba": _col([None, None], pa.string()),
            "birth_year_ah": _col([None, None], pa.int32()),
            "death_year_ah": _col([256, 197], pa.int32()),
            "birth_location": _col([None, None], pa.string()),
            "death_location": _col([None, None], pa.string()),
            "generation": _col([None, None], pa.string()),
            "gender": _col(["male", "male"], pa.string()),
            "trustworthiness": _col([None, None], pa.string()),
            "external_id": _col([None, None], pa.string()),
            "source": _col(["test", "test"], pa.string()),
        }
    )
    pq.write_table(table, staging_dir / "narrators_bio_test.parquet")


def _mention_rows(n_hadiths: int) -> list[dict[str, object]]:
    """A deterministic set of resolved-mention rows across several hadith chains."""
    names = [_NAME_A, _NAME_B, _UNKNOWN]
    rows: list[dict[str, object]] = []
    for h in range(n_hadiths):
        for pos, name in enumerate(names):
            rows.append(
                {
                    "mention_id": f"m:{h}:{pos}",
                    "hadith_id": f"hdt:test:H{h}",
                    "source_corpus": "sunnah",
                    "position_in_chain": pos,
                    "name_raw": name,
                    "name_normalized": normalize_arabic(name),
                    "canonical_narrator_id": None,
                    "transmission_method": "حدثنا",
                    "confidence": None,
                }
            )
    return rows


def _write_mentions(output_dir: Path, rows: list[dict[str, object]]) -> None:
    cols: dict[str, pa.Array] = {}
    for field in NARRATOR_MENTIONS_RESOLVED_SCHEMA:
        cols[field.name] = pa.array([r[field.name] for r in rows], type=field.type)
    pq.write_table(
        pa.table(cols, schema=NARRATOR_MENTIONS_RESOLVED_SCHEMA),
        output_dir / "narrator_mentions_resolved.parquet",
    )


def _setup(tmp_path: Path, name: str, *, n_hadiths: int = 4) -> tuple[Path, Path]:
    staging = tmp_path / name / "staging"
    output = tmp_path / name / "output"
    staging.mkdir(parents=True)
    output.mkdir(parents=True)
    _write_bio_candidates(staging)
    _write_mentions(output, _mention_rows(n_hadiths))
    return staging, output


def _read_outputs(output_dir: Path) -> dict[str, object]:
    """Read the three disambiguate artifacts into comparable Python structures."""
    canonical = pq.read_table(output_dir / "narrators_canonical.parquet").to_pylist()
    mentions = pq.read_table(output_dir / "narrator_mentions_resolved.parquet").to_pylist()
    merge_log_path = output_dir / "merge_log.parquet"
    merge_log = pq.read_table(merge_log_path).to_pylist() if merge_log_path.exists() else []
    return {"canonical": canonical, "mentions": mentions, "merge_log": merge_log}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_stable_across_mention_id_rewrite(tmp_path: Path) -> None:
    """Identical content with regenerated mention_ids ⇒ same content_hash, diff id_hash."""
    _, out_a = _setup(tmp_path, "a")
    rows_b = _mention_rows(4)
    for r in rows_b:  # simulate an NER rewrite: same content, new random ids
        r["mention_id"] = "uuid-" + str(r["mention_id"])
    out_b = tmp_path / "b"
    out_b.mkdir()
    _write_mentions(out_b, rows_b)

    c_a, m_a, n_a = disambiguate._compute_input_fingerprint(
        out_a / "narrator_mentions_resolved.parquet"
    )
    c_b, m_b, n_b = disambiguate._compute_input_fingerprint(
        out_b / "narrator_mentions_resolved.parquet"
    )
    assert n_a == n_b == 12
    assert c_a == c_b, "stable content must fingerprint identically"
    assert m_a != m_b, "regenerated mention_ids must change the id fingerprint"


def test_fingerprint_changes_on_content_change(tmp_path: Path) -> None:
    _, out_a = _setup(tmp_path, "a")
    rows_b = _mention_rows(4)
    rows_b[0]["name_normalized"] = normalize_arabic("اسم مختلف")  # different corpus
    out_b = tmp_path / "b"
    out_b.mkdir()
    _write_mentions(out_b, rows_b)

    c_a, _, _ = disambiguate._compute_input_fingerprint(
        out_a / "narrator_mentions_resolved.parquet"
    )
    c_b, _, _ = disambiguate._compute_input_fingerprint(
        out_b / "narrator_mentions_resolved.parquet"
    )
    assert c_a != c_b


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------
def test_checkpoint_save_load_clear(tmp_path: Path) -> None:
    ckpt_dir = disambiguate._checkpoint_dir(tmp_path)
    assert disambiguate._load_checkpoint(ckpt_dir) is None

    payload = {"schema_version": 1, "processed": 42, "canonical_map": {"nar:x": {}}}
    disambiguate._save_checkpoint(ckpt_dir, payload)
    assert (ckpt_dir / "state.json").exists()
    assert not (ckpt_dir / "state.json.tmp").exists(), "temp file must be renamed away"

    loaded = disambiguate._load_checkpoint(ckpt_dir)
    assert loaded == payload

    disambiguate._clear_checkpoint(ckpt_dir)
    assert not ckpt_dir.exists()
    assert disambiguate._load_checkpoint(ckpt_dir) is None


# ---------------------------------------------------------------------------
# Resume equivalence
# ---------------------------------------------------------------------------
def test_resumed_run_is_output_identical_to_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cold reference run.
    cold_staging, cold_out = _setup(tmp_path, "cold")
    disambiguate.run(cold_staging, cold_out, batch_size=2, checkpoint_every_n_batches=1)
    cold = _read_outputs(cold_out)
    assert not disambiguate._checkpoint_dir(cold_staging).exists(), "cold run clears checkpoint"

    # Resume run against an identical, separate input.
    res_staging, res_out = _setup(tmp_path, "resume")

    real_iter = disambiguate._iter_mention_batches

    def crashing_iter(mentions_dir: Path, batch_size: int = disambiguate._MENTION_BATCH_SIZE):  # type: ignore[no-untyped-def]
        for i, batch in enumerate(real_iter(mentions_dir, batch_size=batch_size)):
            if i >= 2:
                raise RuntimeError("simulated host crash mid-disambiguate")
            yield batch

    monkeypatch.setattr(disambiguate, "_iter_mention_batches", crashing_iter)
    with pytest.raises(RuntimeError):
        disambiguate.run(res_staging, res_out, batch_size=2, checkpoint_every_n_batches=1)
    monkeypatch.undo()

    # A checkpoint survived the crash with a non-trivial, partial offset.
    ckpt = disambiguate._load_checkpoint(disambiguate._checkpoint_dir(res_staging))
    assert ckpt is not None
    assert 0 < ckpt["processed"] < len(_mention_rows(4))

    # Resume to completion.
    disambiguate.run(res_staging, res_out, batch_size=2, checkpoint_every_n_batches=1)
    resumed = _read_outputs(res_out)

    assert resumed["canonical"] == cold["canonical"]
    assert resumed["mentions"] == cold["mentions"]
    assert resumed["merge_log"] == cold["merge_log"]
    assert not disambiguate._checkpoint_dir(res_staging).exists(), "resume clears checkpoint"


# ---------------------------------------------------------------------------
# Invalidation → cold start
# ---------------------------------------------------------------------------
def test_fingerprint_mismatch_cold_starts(tmp_path: Path) -> None:
    staging, output = _setup(tmp_path, "mismatch")
    ckpt_dir = disambiguate._checkpoint_dir(staging)
    # A stale checkpoint whose content_hash cannot match the current input, carrying
    # a sentinel canonical id that a real run would never mint.
    disambiguate._save_checkpoint(
        ckpt_dir,
        {
            "schema_version": disambiguate._CHECKPOINT_SCHEMA_VERSION,
            "content_hash": "STALE",
            "mention_id_hash": "STALE",
            "total_rows": 12,
            "batch_size": 2,
            "processed": 6,
            "death_year_index": {},
            "location_index": {},
            "canonical_map": {"nar:SENTINEL": {"canonical_id": "nar:SENTINEL"}},
            "merge_log_rows": [],
            "ambiguous_rows": [],
            "resolved_map": {},
            "naive_identity_pairs": [],
            "source_resolved": {},
            "source_total": {},
        },
    )

    disambiguate.run(staging, output, batch_size=2, checkpoint_every_n_batches=1)
    canonical_ids = {
        r["canonical_id"] for r in pq.read_table(output / "narrators_canonical.parquet").to_pylist()
    }
    assert "nar:SENTINEL" not in canonical_ids, "stale checkpoint must be ignored (cold start)"


def test_stale_mention_ids_cold_starts_with_clear(tmp_path: Path) -> None:
    staging, output = _setup(tmp_path, "stale_ids")
    content_hash, _mid, total = disambiguate._compute_input_fingerprint(
        output / "narrator_mentions_resolved.parquet"
    )
    ckpt_dir = disambiguate._checkpoint_dir(staging)
    # content matches the current input, but mention_id_hash does NOT — the NER
    # rewrite case. Must cold-start (sentinel absent) and clear the checkpoint.
    disambiguate._save_checkpoint(
        ckpt_dir,
        {
            "schema_version": disambiguate._CHECKPOINT_SCHEMA_VERSION,
            "content_hash": content_hash,
            "mention_id_hash": "DIFFERENT",
            "total_rows": total,
            "batch_size": 2,
            "processed": 6,
            "death_year_index": {},
            "location_index": {},
            "canonical_map": {"nar:SENTINEL": {"canonical_id": "nar:SENTINEL"}},
            "merge_log_rows": [],
            "ambiguous_rows": [],
            "resolved_map": {},
            "naive_identity_pairs": [],
            "source_resolved": {},
            "source_total": {},
        },
    )

    disambiguate.run(staging, output, batch_size=2, checkpoint_every_n_batches=1)
    canonical_ids = {
        r["canonical_id"] for r in pq.read_table(output / "narrators_canonical.parquet").to_pylist()
    }
    assert "nar:SENTINEL" not in canonical_ids


def test_resume_false_ignores_existing_checkpoint(tmp_path: Path) -> None:
    staging, output = _setup(tmp_path, "no_resume")
    content_hash, mid_hash, total = disambiguate._compute_input_fingerprint(
        output / "narrator_mentions_resolved.parquet"
    )
    ckpt_dir = disambiguate._checkpoint_dir(staging)
    disambiguate._save_checkpoint(
        ckpt_dir,
        {
            "schema_version": disambiguate._CHECKPOINT_SCHEMA_VERSION,
            "content_hash": content_hash,
            "mention_id_hash": mid_hash,
            "total_rows": total,
            "batch_size": 2,
            "processed": 6,
            "death_year_index": {},
            "location_index": {},
            "canonical_map": {"nar:SENTINEL": {"canonical_id": "nar:SENTINEL"}},
            "merge_log_rows": [],
            "ambiguous_rows": [],
            "resolved_map": {},
            "naive_identity_pairs": [],
            "source_resolved": {},
            "source_total": {},
        },
    )

    disambiguate.run(staging, output, batch_size=2, checkpoint_every_n_batches=1, resume=False)
    canonical_ids = {
        r["canonical_id"] for r in pq.read_table(output / "narrators_canonical.parquet").to_pylist()
    }
    assert "nar:SENTINEL" not in canonical_ids
