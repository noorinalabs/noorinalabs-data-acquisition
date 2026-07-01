"""Tests for the parallelized (process-pool) embedding encode (da#246).

The parallel path must be a drop-in for the serial path: identical embeddings,
identical row ordering, and the same crash-resume semantics — only faster. These
tests exercise that equivalence with a deterministic fake model (no ML deps) and
the ``fork`` start method so they run fast and hermetically in CI.

``FakeModel`` and ``_fake_provider`` are module-level so they remain importable /
picklable under any multiprocessing start method.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.resolve.dedup import (
    _MODEL_NAME,
    _encode_with_resume,
    _resolve_encode_workers,
)


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Encodes each text to ``[len(text)] * dim`` so a caller can check both the
    values and their placement, and counts encodes so a test can prove which
    path (serial parent-model vs parallel per-worker model) actually ran.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encoded = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, batch: list[str], **_: object) -> np.ndarray:
        self.encoded += len(batch)
        return np.array([[float(len(t))] * self.dim for t in batch], dtype=np.float32)


def _fake_provider() -> FakeModel:
    """Per-worker model factory (module-level so it is picklable for spawn)."""
    return FakeModel()


def _ids(n: int) -> list[str]:
    return [f"h{i}" for i in range(n)]


def _texts(n: int) -> list[str]:
    # Distinct lengths per index so emb[i, 0] == len(texts[i]) is a placement check.
    return ["x" * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Determinism: parallel output == serial output
# ---------------------------------------------------------------------------
def test_parallel_matches_serial_exactly(tmp_path: Path) -> None:
    texts = _texts(37)
    ids = _ids(37)

    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    serial = _encode_with_resume(texts, ids, serial_dir, FakeModel(), batch_size=4, workers=1)

    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()
    parent_model = FakeModel()
    parallel = _encode_with_resume(
        texts,
        ids,
        parallel_dir,
        parent_model,
        batch_size=4,
        workers=3,
        model_provider=_fake_provider,
        mp_start_method="fork",
    )

    assert np.array_equal(np.asarray(serial), np.asarray(parallel))
    # The parallel path must NOT have used the parent model — the workers did.
    assert parent_model.encoded == 0
    # And placement is stable: row i holds len(texts[i]).
    assert [int(parallel[i, 0]) for i in range(37)] == [i + 1 for i in range(37)]


def test_parallel_ordering_stable_across_chunks(tmp_path: Path) -> None:
    # More rows than a single chunk forces multiple out-of-order completions.
    texts = _texts(100)
    ids = _ids(100)
    emb = _encode_with_resume(
        texts,
        ids,
        tmp_path,
        FakeModel(),
        batch_size=8,
        workers=4,
        model_provider=_fake_provider,
        mp_start_method="fork",
    )
    assert emb.shape == (100, 4)
    assert [int(emb[i, 0]) for i in range(100)] == [i + 1 for i in range(100)]
    assert (tmp_path / "hadith_embeddings.progress").read_text().strip() == "100"


# ---------------------------------------------------------------------------
# Resume semantics carry over to the parallel path
# ---------------------------------------------------------------------------
def test_parallel_resume_skips_completed_prefix(tmp_path: Path) -> None:
    texts = _texts(20)
    ids = _ids(20)
    # Simulate a crash after 6 of 20 rows.
    import hashlib

    mm = np.lib.format.open_memmap(
        tmp_path / "hadith_embeddings.npy", mode="w+", dtype=np.float32, shape=(20, 4)
    )
    for i in range(6):
        mm[i] = float(i + 1)
    mm.flush()
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    (tmp_path / "hadith_embeddings.meta.json").write_text(
        json.dumps({"count": 20, "dim": 4, "model": _MODEL_NAME, "ids_hash": ids_hash})
    )
    (tmp_path / "hadith_embeddings.progress").write_text("6")

    emb = _encode_with_resume(
        texts,
        ids,
        tmp_path,
        FakeModel(),
        batch_size=4,
        workers=3,
        model_provider=_fake_provider,
        mp_start_method="fork",
    )
    # Preserved prefix + newly-encoded remainder, all correctly placed.
    assert [int(emb[i, 0]) for i in range(20)] == [i + 1 for i in range(20)]


# ---------------------------------------------------------------------------
# Serial fallback
# ---------------------------------------------------------------------------
def test_workers_one_uses_serial_path(tmp_path: Path) -> None:
    texts = _texts(10)
    model = FakeModel()
    _encode_with_resume(
        texts, _ids(10), tmp_path, model, batch_size=4, workers=1, model_provider=_fake_provider
    )
    # Serial path encodes via the passed-in parent model.
    assert model.encoded == 10


def test_small_corpus_falls_back_to_serial(tmp_path: Path) -> None:
    # remaining (3) < 2 * batch_size (4) -> not worth a process pool.
    texts = _texts(3)
    model = FakeModel()
    _encode_with_resume(
        texts,
        _ids(3),
        tmp_path,
        model,
        batch_size=2,
        workers=4,
        model_provider=_fake_provider,
        mp_start_method="fork",
    )
    assert model.encoded == 3


def test_missing_provider_falls_back_to_serial(tmp_path: Path) -> None:
    texts = _texts(20)
    model = FakeModel()
    _encode_with_resume(
        texts, _ids(20), tmp_path, model, batch_size=4, workers=4, model_provider=None
    )
    assert model.encoded == 20


# ---------------------------------------------------------------------------
# Worker-count resolution + oversubscription guard
# ---------------------------------------------------------------------------
@patch("src.resolve.dedup.os.cpu_count", return_value=16)
def test_resolve_auto_caps_at_auto_cap(_cpu: object) -> None:
    from src.resolve.dedup import _AUTO_WORKER_CAP

    assert _resolve_encode_workers(0, 10_000) == _AUTO_WORKER_CAP
    assert _resolve_encode_workers(None, 10_000) == _AUTO_WORKER_CAP


@patch("src.resolve.dedup.os.cpu_count", return_value=4)
def test_resolve_explicit_request_clamped_to_cores(_cpu: object) -> None:
    # Explicit over-request is honoured only up to the core count (no oversub).
    assert _resolve_encode_workers(100, 10_000) == 4
    # An in-range request is honoured verbatim.
    assert _resolve_encode_workers(2, 10_000) == 2


@patch("src.resolve.dedup.os.cpu_count", return_value=8)
def test_resolve_never_exceeds_item_count(_cpu: object) -> None:
    assert _resolve_encode_workers(0, 3) == 3
    assert _resolve_encode_workers(0, 0) == 1


@patch("src.resolve.dedup.os.cpu_count", return_value=1)
def test_resolve_single_core_is_serial(_cpu: object) -> None:
    assert _resolve_encode_workers(0, 10_000) == 1
    assert _resolve_encode_workers(8, 10_000) == 1
