"""Crash-resume tests for dedup's FAISS search + pair-collection phase (da#272).

The embedding encode is already resumable (da#245); this covers the phase AFTER
it. The resumable loop is exercised through :func:`_search_and_collect_resumable`
with a numpy-only fake index, so it runs in CI without faiss / sentence-
transformers while still asserting real behaviour: a resumed collection is
output-identical to a cold one, a stale-fingerprint checkpoint is discarded, and
``resume=False`` cold-starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src.resolve import _checkpoint
from src.resolve.dedup import _search_and_collect_resumable

# 4 exact-duplicate embedding pairs (one-hot per pair), each an even (sunni) +
# odd (shia) row, so inner product is 1.0 within a pair and 0.0 across pairs.
_EMB = np.array(
    [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
_IDS = [f"h{i}" for i in range(8)]
_ID_TO_CORPUS = {f"h{i}": ("sunnah" if i % 2 == 0 else "thaqalayn") for i in range(8)}
_ACTUAL_K = 8
_THRESHOLD = 0.5


class FakeIndex:
    """Deterministic IndexFlatIP stand-in: top-k neighbours by inner product."""

    def __init__(self, emb: npt.NDArray[np.float32]) -> None:
        self.emb = emb

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        sims = queries @ self.emb.T
        idx = np.argsort(-sims, axis=1)[:, :k].astype(np.int64)
        scores = np.take_along_axis(sims, idx, axis=1).astype(np.float32)
        return scores, idx


class CrashingIndex(FakeIndex):
    """Fake index that raises after ``crash_after`` search calls (a host crash)."""

    def __init__(self, emb: npt.NDArray[np.float32], crash_after: int) -> None:
        super().__init__(emb)
        self.calls = 0
        self.crash_after = crash_after

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("simulated host crash mid-dedup-collection")
        return super().search(queries, k)


def _collect(
    ckpt_dir: Path,
    fingerprint: str,
    *,
    resume: bool,
    index_available: bool,
    build_index: object,
    reload_index: object,
    block_size: int = 2,
    cadence: int = 1,
) -> tuple[list[str], list[str], list[float], list[str], list[bool]]:
    return _search_and_collect_resumable(
        embeddings=_EMB,
        hadith_ids=_IDS,
        id_to_corpus=_ID_TO_CORPUS,
        actual_k=_ACTUAL_K,
        threshold=_THRESHOLD,
        ckpt_dir=ckpt_dir,
        fingerprint=fingerprint,
        cadence=cadence,
        resume=resume,
        index_available=index_available,
        build_index=build_index,  # type: ignore[arg-type]
        reload_index=reload_index,  # type: ignore[arg-type]
        block_size=block_size,
    )


def _cold(ckpt_dir: Path, fingerprint: str = "fp") -> tuple[list[str], ...]:
    idx = FakeIndex(_EMB)
    return _collect(
        ckpt_dir,
        fingerprint,
        resume=True,
        index_available=False,
        build_index=lambda: idx,
        reload_index=lambda: idx,
        block_size=8,  # single block, whole corpus
    )


def test_cold_collection_finds_cross_sect_pairs(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "dedup")
    ids_a, ids_b, scores, variants, cross = _cold(ckpt_dir)
    assert list(zip(ids_a, ids_b)) == [("h0", "h1"), ("h2", "h3"), ("h4", "h5"), ("h6", "h7")]
    assert all(cross), "even=sunni + odd=shia ⇒ every pair is cross-sect"
    assert all(s == pytest.approx(1.0) for s in scores)


def test_resumed_collection_is_output_identical_to_cold(tmp_path: Path) -> None:
    cold = _cold(_checkpoint.checkpoint_dir(tmp_path / "cold", "dedup"))

    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path / "resume", "dedup")
    # Crash after 2 of the 4 blocks: blocks 0,1 checkpoint (processed=4), block 2 dies.
    crashing = CrashingIndex(_EMB, crash_after=2)
    with pytest.raises(RuntimeError):
        _collect(
            ckpt_dir,
            "fp",
            resume=True,
            index_available=False,
            build_index=lambda: crashing,
            reload_index=lambda: crashing,
        )

    saved = _checkpoint.load_checkpoint(ckpt_dir)
    assert saved is not None
    assert saved["processed_rows"] == 4, "checkpoint lands on the last completed block boundary"

    # Resume with a healthy index present — must reproduce the cold output exactly.
    fresh = FakeIndex(_EMB)
    reloaded = {"used": False}

    def _reload() -> FakeIndex:
        reloaded["used"] = True
        return fresh

    resumed = _collect(
        ckpt_dir,
        "fp",
        resume=True,
        index_available=True,
        build_index=lambda: pytest.fail("resume must reload, not rebuild"),
        reload_index=_reload,
    )
    assert reloaded["used"], "a valid resume must reload the persisted index"
    assert resumed == cold


def test_stale_fingerprint_cold_starts(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "dedup")
    _checkpoint.save_checkpoint(
        ckpt_dir,
        {
            "schema_version": 1,
            "fingerprint": "STALE",
            "processed_rows": 4,
            "seen_pairs": [["SENTINEL_A", "SENTINEL_B"]],
            "ids_a": ["SENTINEL_A"],
            "ids_b": ["SENTINEL_B"],
            "sim_scores": [1.0],
            "variant_types": ["verbatim"],
            "cross_sects": [True],
        },
    )
    idx = FakeIndex(_EMB)
    ids_a, *_ = _collect(
        ckpt_dir,
        "fp-current",  # != "STALE"
        resume=True,
        index_available=True,
        build_index=lambda: idx,
        reload_index=lambda: pytest.fail("stale fingerprint must cold-start (build, not reload)"),
    )
    assert "SENTINEL_A" not in ids_a, "stale checkpoint must be discarded"


def test_resume_false_ignores_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = _checkpoint.checkpoint_dir(tmp_path, "dedup")
    _checkpoint.save_checkpoint(
        ckpt_dir,
        {
            "schema_version": 1,
            "fingerprint": "fp",  # would otherwise match
            "processed_rows": 8,
            "seen_pairs": [["SENTINEL_A", "SENTINEL_B"]],
            "ids_a": ["SENTINEL_A"],
            "ids_b": ["SENTINEL_B"],
            "sim_scores": [1.0],
            "variant_types": ["verbatim"],
            "cross_sects": [True],
        },
    )
    idx = FakeIndex(_EMB)
    ids_a, *_ = _collect(
        ckpt_dir,
        "fp",
        resume=False,
        index_available=True,
        build_index=lambda: idx,
        reload_index=lambda: pytest.fail("resume=False must cold-start"),
    )
    assert "SENTINEL_A" not in ids_a, "resume=False must ignore even a matching checkpoint"
