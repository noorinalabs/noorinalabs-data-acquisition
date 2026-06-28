"""Tests for the crash-resumable, memory-bounded embedding encode (da#245)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.resolve.dedup import _MODEL_NAME, _encode_with_resume


class FakeModel:
    """Deterministic stand-in for SentenceTransformer (no ML deps in the test).

    Encodes each text to a fixed-dim vector and counts how many texts it has
    encoded, so a test can assert that a resumed run does NOT re-encode.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encoded = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, batch: list[str], **_: object) -> np.ndarray:
        self.encoded += len(batch)
        # value = string length, broadcast across dims — deterministic + checkable.
        return np.array([[float(len(t))] * self.dim for t in batch], dtype=np.float32)


def _ids(n: int) -> list[str]:
    return [f"h{i}" for i in range(n)]


def test_encodes_all_and_writes_memmap(tmp_path: Path) -> None:
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    model = FakeModel()
    emb = _encode_with_resume(texts, _ids(len(texts)), tmp_path, model, batch_size=2)
    assert emb.shape == (5, 4)
    assert model.encoded == 5
    # values are the text lengths
    assert [int(emb[i, 0]) for i in range(5)] == [1, 2, 3, 4, 5]
    # memmap + sidecars persisted
    assert (tmp_path / "hadith_embeddings.npy").exists()
    assert (tmp_path / "hadith_embeddings.progress").read_text().strip() == "5"
    meta = json.loads((tmp_path / "hadith_embeddings.meta.json").read_text())
    assert meta == {"count": 5, "dim": 4, "model": _MODEL_NAME, "ids_hash": meta["ids_hash"]}


def test_resume_skips_already_encoded(tmp_path: Path) -> None:
    texts = ["a", "bb", "ccc", "dddd"]
    ids = _ids(len(texts))
    # First (complete) run.
    _encode_with_resume(texts, ids, tmp_path, FakeModel(), batch_size=2)
    # Second run on the SAME corpus must re-encode nothing.
    model2 = FakeModel()
    emb = _encode_with_resume(texts, ids, tmp_path, model2, batch_size=2)
    assert model2.encoded == 0
    assert [int(emb[i, 0]) for i in range(4)] == [1, 2, 3, 4]


def test_partial_progress_resumes_from_offset(tmp_path: Path) -> None:
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    ids = _ids(len(texts))
    # Simulate a crash after 2 of 6 rows: create the memmap + meta + progress=2.
    dim = 4
    mm = np.lib.format.open_memmap(
        tmp_path / "hadith_embeddings.npy", mode="w+", dtype=np.float32, shape=(6, dim)
    )
    mm[0] = 1.0
    mm[1] = 2.0
    mm.flush()
    import hashlib

    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    (tmp_path / "hadith_embeddings.meta.json").write_text(
        json.dumps({"count": 6, "dim": dim, "model": _MODEL_NAME, "ids_hash": ids_hash})
    )
    (tmp_path / "hadith_embeddings.progress").write_text("2")

    model = FakeModel()
    emb = _encode_with_resume(texts, ids, tmp_path, model, batch_size=2)
    # Only the remaining 4 rows re-encoded; the first 2 preserved as-is.
    assert model.encoded == 4
    assert int(emb[0, 0]) == 1 and int(emb[1, 0]) == 2
    assert [int(emb[i, 0]) for i in range(2, 6)] == [3, 4, 5, 6]


def test_meta_mismatch_forces_full_reencode(tmp_path: Path) -> None:
    # A stale memmap from a DIFFERENT corpus must not be reused.
    _encode_with_resume(["a", "bb"], _ids(2), tmp_path, FakeModel(), batch_size=2)
    # Different corpus (different ids/count) → full re-encode.
    model = FakeModel()
    emb = _encode_with_resume(["x", "yy", "zzz"], _ids(3), tmp_path, model, batch_size=2)
    assert emb.shape == (3, 4)
    assert model.encoded == 3


@pytest.mark.parametrize("batch_size", [1, 2, 3, 10])
def test_batch_size_invariant(tmp_path: Path, batch_size: int) -> None:
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    emb = _encode_with_resume(texts, _ids(len(texts)), tmp_path, FakeModel(), batch_size=batch_size)
    assert [int(emb[i, 0]) for i in range(5)] == [1, 2, 3, 4, 5]
