"""Hadith deduplication and parallel detection.

Generates sentence-transformer embeddings for hadith matn texts, builds a FAISS
index, and identifies parallel hadith pairs across collections and sects.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from src.models.enums import VariantType
from src.resolve._checkpoint import (
    CheckpointController,
    checkpoint_dir,
    clear_checkpoint,
    hash_strings,
    load_checkpoint,
    log_resume,
    resolve_cadence,
    save_checkpoint,
)
from src.resolve._deps import MissingDependencyError, missing_dependencies
from src.resolve._provenance import (
    DetectorProvenance,
    DetectorStatus,
    write_parallel_links,
)
from src.resolve.schemas import PARALLEL_LINKS_SCHEMA
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    import faiss as faiss_mod
    import numpy as np
    import numpy.typing as npt

    class _Searchable(Protocol):
        """Anything with a FAISS-style block search (a real ``faiss.Index`` or a
        test double). Decouples the resumable collection loop from faiss so it is
        unit-testable without the native dep."""

        def search(
            self, queries: npt.NDArray[np.float32], k: int
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...


logger = get_logger(__name__)

__all__ = ["run", "run_dedup"]

# Sentence-transformer used for hadith-matn embeddings. Pinned name so the resume
# meta can refuse to reuse embeddings produced by a different model.
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Ceiling for the *auto* (unset / non-positive) encode-worker count. Each worker
# holds its own copy of the ~0.5 GB sentence-transformer, so more processes than
# this trade RAM for throughput we do not need; an explicit request may exceed it
# but is still capped at the physical core count (see `_resolve_encode_workers`).
_AUTO_WORKER_CAP = 8

# Split each worker's share into several chunks so a straggler chunk cannot
# leave one core idle while the others finish — cheap load balancing.
_CHUNKS_PER_WORKER = 4

# Crash-resume for the FAISS search + pair-collection phase (da#272). The encode
# is already resumable via the embedding memmap (da#245); this checkpoints the
# stage AFTER it. Queries are searched in fixed row-blocks so a crash costs at
# most `_DEDUP_CHECKPOINT_EVERY_N_BLOCKS` blocks of pair collection, and the
# persisted FAISS index is reloaded on resume so an IVF index's random kmeans
# init cannot make the resumed search diverge from the crashed run's.
_DEDUP_SEARCH_BLOCK = 10_000
_DEDUP_CHECKPOINT_EVERY_N_BLOCKS = 4
# v2 (da#321): stored `cross_sects` are now derived from the authoritative
# `sect` column, not the corpus-name allowlist — bump so a checkpoint written by
# the old formula is discarded rather than restored with stale cross_sect values.
_DEDUP_CHECKPOINT_SCHEMA_VERSION = 2
_FAISS_INDEX_FILENAME = "hadith_embeddings.faiss"

# Staging inputs this stage consumes. Note the plural `hadiths_` prefix.
_HADITH_GLOB = "**/hadiths_*.parquet"

# Third-party modules this stage declares (the `ml` dependency group, plus numpy,
# which arrives transitively). Absence of any of these on an enabled stage is an
# environment defect, not an empty result — see src/resolve/_deps.py (da#309).
_DECLARED_DEPENDENCIES = ("numpy", "sentence_transformers", "faiss")


def _id_set_hash(hadith_ids: list[str]) -> str:
    """Stable SHA-256 over the corpus id list in order.

    The resume-identity of a dedup run: the same ids in the same order yield the
    same embeddings and therefore the same pair set. Matches the ``ids_hash`` the
    encode memmap records (:func:`_encode_with_resume`), so the two resume layers
    agree on what "same corpus" means.
    """
    return hashlib.sha256("\n".join(hadith_ids).encode("utf-8")).hexdigest()


def _classify_pair(score: float) -> VariantType:
    """Classify a similarity score into a variant type tier."""
    if score >= 0.90:
        return VariantType.VERBATIM
    if score >= 0.80:
        return VariantType.CLOSE_PARAPHRASE
    return VariantType.THEMATIC


def _is_cross_sect(sect_a: str | None, sect_b: str | None) -> bool:
    """Return True when two hadiths carry different (non-empty) ``sect`` labels.

    Reads the authoritative hadith ``sect`` column — the same signal the
    deterministic lexical detector (``parallels.py``) uses. The two detectors
    both materialize ``parallel_links.parquet`` and ``run_all`` composes them,
    letting a *semantic* row overwrite the deterministic one on a shared
    canonical pair; if the two derive ``cross_sect`` from different sources the
    composed value flips non-deterministically with the embedder environment.

    Pre-da#321 this inferred sect from the corpus *name* via hard-coded
    sunni/shia source allowlists, which (a) mislabeled any corpus absent from
    the lists as non-cross-sect and (b) disagreed with the authoritative
    ``sect`` column ``parallels.py`` reads — so a semantic row could clobber a
    correct cross-sect edge with ``False``. Deriving from ``sect`` here makes
    both detectors agree. See da#321.
    """
    return bool(sect_a and sect_b and sect_a != sect_b)


def _hadith_files(staging_dir: Path) -> list[Path]:
    """Return the staging hadith Parquet files this stage consumes, sorted."""
    return sorted(staging_dir.glob(_HADITH_GLOB))


def _load_hadith_texts(
    staging_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Load hadith IDs, English matn texts, and ``sect`` labels from staging Parquets.

    Returns (hadith_ids, texts, sects) with null/empty matn_en rows excluded. The
    third element is the authoritative ``sect`` column, used to classify a pair as
    cross-sect (see :func:`_is_cross_sect` / da#321) — not the source corpus.
    """
    hadith_files = _hadith_files(staging_dir)
    if not hadith_files:
        logger.warning("dedup_no_hadith_files", staging_dir=str(staging_dir))
        return [], [], []

    ids: list[str] = []
    texts: list[str] = []
    sects: list[str] = []
    skipped = 0
    for fpath in hadith_files:
        table = pq.read_table(fpath, columns=["source_id", "matn_en", "sect"])
        for i in range(table.num_rows):
            matn = table.column("matn_en")[i].as_py()
            if not matn or not matn.strip():
                skipped += 1
                continue
            ids.append(table.column("source_id")[i].as_py())
            texts.append(matn)
            sects.append(table.column("sect")[i].as_py())

    logger.info(
        "dedup_loaded_hadiths",
        included=len(ids),
        skipped=skipped,
        files=len(hadith_files),
    )
    return ids, texts, sects


def _build_default_model() -> object:
    """Construct the pinned sentence-transformer.

    Used both by the serial path and — as the per-worker ``model_provider`` — by
    each parallel encode worker, so every process builds its own model from the
    same pinned name (models are not fork/pickle-safe to share across processes).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def _resolve_encode_workers(requested: int | None, n_items: int) -> int:
    """Resolve the effective encode-worker count, guarding against oversubscription.

    ``requested`` is the configured value (``DEDUP_ENCODE_WORKERS`` / the
    ``encode_workers`` argument): ``None`` or ``<= 0`` means *auto* (scale to the
    box, capped at :data:`_AUTO_WORKER_CAP`); a positive value is honoured but
    still clamped to the physical core count so we never spin more CPU-bound
    embedding processes than there are cores. Never returns less than 1.
    """
    cpu = os.cpu_count() or 1
    if requested is None or requested <= 0:
        workers = min(cpu, _AUTO_WORKER_CAP)
    else:
        workers = min(requested, cpu)
    # Nothing to gain from more workers than items.
    workers = min(workers, max(1, n_items))
    return max(1, workers)


# --- Parallel-encode worker plumbing --------------------------------------
# Each worker process holds one model in a module global (populated by the pool
# initializer) so the model is built once per worker rather than once per chunk.
_WORKER_MODEL: object | None = None


def _set_thread_limit(thread_limit: int) -> None:
    """Pin per-process BLAS/torch intra-op threads.

    Without this, every worker's torch/OpenBLAS would each try to use all cores,
    so ``workers × cores`` threads would thrash a single machine. One thread per
    worker keeps the fan-out to exactly ``workers`` busy cores.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(thread_limit)
    try:
        import torch

        torch.set_num_threads(thread_limit)
    except Exception:  # noqa: BLE001  (torch optional / already-configured — best effort)
        pass


def _worker_init(model_provider: Callable[[], object], thread_limit: int) -> None:
    """Pool initializer: pin threads and build this worker's model once."""
    global _WORKER_MODEL
    _set_thread_limit(thread_limit)
    _WORKER_MODEL = model_provider()


def _worker_encode_chunk(
    task: tuple[int, list[str], int],
) -> tuple[int, npt.NDArray[np.float32]]:
    """Encode one contiguous chunk of texts and return ``(start_offset, embeddings)``.

    Runs in a worker process. The returned ``start_offset`` lets the parent place
    the block at a fixed row range regardless of completion order — this is what
    makes the parallel output byte-identical to the serial output.
    """
    import numpy as np

    start_offset, batch_texts, batch_size = task
    model = _WORKER_MODEL
    assert model is not None, "worker model not initialized"
    parts: list[npt.NDArray[np.float32]] = []
    for s in range(0, len(batch_texts), batch_size):
        parts.append(
            model.encode(  # type: ignore[attr-defined]
                batch_texts[s : s + batch_size],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
    return start_offset, np.vstack(parts).astype(np.float32)


def _encode_range_serial(
    texts: list[str],
    emb: npt.NDArray[np.float32],
    start: int,
    n: int,
    batch_size: int,
    prog_path: Path,
    model: object,
) -> None:
    """Serial fallback: encode ``[start, n)`` in-process, one batch at a time."""
    import resource

    t0 = time.monotonic()
    for s in range(start, n, batch_size):
        e = min(s + batch_size, n)
        emb[s:e] = model.encode(  # type: ignore[attr-defined]
            texts[s:e],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        prog_path.write_text(str(e))
        elapsed = time.monotonic() - t0
        rate = (e - start) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "dedup_encoding_progress",
            processed=e,
            total=n,
            pct=round(e / n * 100, 1),
            rate_per_s=round(rate, 1),
            eta_s=round((n - e) / rate) if rate > 0 else None,
            rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        )


def _encode_range_parallel(
    texts: list[str],
    emb: npt.NDArray[np.float32],
    start: int,
    n: int,
    batch_size: int,
    workers: int,
    model_provider: Callable[[], object],
    prog_path: Path,
    mp_start_method: str,
) -> None:
    """Encode ``[start, n)`` across a process pool, writing each block in place.

    The remaining rows are split into ``workers × _CHUNKS_PER_WORKER`` chunks;
    each worker encodes a chunk and returns its block, which the parent writes to
    the fixed memmap slice ``emb[cs:ce]``. Because placement is by row index, not
    completion order, the result is identical to the serial encode. The resume
    marker advances only over the *contiguous* completed prefix, so a crash
    mid-fan-out resumes correctly (any already-written non-contiguous block is
    simply, and idempotently, re-encoded).
    """
    import resource

    remaining = n - start
    n_chunks = min(remaining, max(1, workers * _CHUNKS_PER_WORKER))
    chunk_size = math.ceil(remaining / n_chunks)
    bounds: list[tuple[int, int]] = []
    s = start
    while s < n:
        e = min(s + chunk_size, n)
        bounds.append((s, e))
        s = e

    logger.info(
        "dedup_encoding_parallel",
        workers=workers,
        chunks=len(bounds),
        chunk_size=chunk_size,
        start_method=mp_start_method,
    )

    ctx = multiprocessing.get_context(mp_start_method)
    completed: dict[int, int] = {}
    contiguous = start
    done = 0
    t0 = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(model_provider, 1),
    ) as pool:
        fut_bounds = {
            pool.submit(_worker_encode_chunk, (cs, texts[cs:ce], batch_size)): (cs, ce)
            for cs, ce in bounds
        }
        for fut in as_completed(fut_bounds):
            cs, ce = fut_bounds[fut]
            _, block = fut.result()
            emb[cs:ce] = block
            completed[cs] = ce
            # Advance the resume marker over the contiguous completed prefix only.
            while contiguous in completed:
                contiguous = completed[contiguous]
            emb.flush()  # type: ignore[attr-defined]  # emb is a np.memmap at runtime
            prog_path.write_text(str(contiguous))
            done += ce - cs
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "dedup_encoding_progress",
                processed=start + done,
                total=n,
                pct=round((start + done) / n * 100, 1),
                rate_per_s=round(rate, 1),
                eta_s=round((remaining - done) / rate) if rate > 0 else None,
                rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                workers=workers,
            )
    prog_path.write_text(str(n))


def _encode_with_resume(
    texts: list[str],
    hadith_ids: list[str],
    staging_dir: Path,
    model: object,
    batch_size: int,
    *,
    workers: int = 1,
    model_provider: Callable[[], object] | None = None,
    mp_start_method: str = "spawn",
) -> npt.NDArray[np.float32]:
    """Encode hadith matn into a disk-backed, crash-resumable embedding memmap.

    Writes embeddings straight into a memory-mapped ``hadith_embeddings.npy`` one
    chunk at a time, so peak RAM is ~one chunk rather than the whole
    ``n × dim`` matrix plus the ``np.vstack`` copy the old path held (da#245
    memory-bounding). A per-chunk progress marker plus a sidecar meta
    (count / dim / model / id-set hash) let a re-run skip already-encoded chunks
    instead of redoing the multi-hour encode after any later-stage crash. Each
    chunk logs throughput, ETA, and an RSS watermark so an approaching OOM shows
    up in the logs rather than as a silent kill.

    When ``workers > 1`` and a ``model_provider`` is supplied, the remaining
    encode is fanned out across a process pool (da#246): each worker builds its
    own model and encodes a chunk, and the parent writes each block to its fixed
    row range so the output is byte-identical to the serial path regardless of
    completion order. ``workers <= 1`` (or too few rows to be worth the process
    overhead) uses the serial fallback. ``mp_start_method`` defaults to ``spawn``
    so a torch-initialized parent does not fork into a deadlock.
    """
    import hashlib
    import json as _json

    import numpy as np

    n = len(texts)
    dim = int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
    emb_path = staging_dir / "hadith_embeddings.npy"
    prog_path = staging_dir / "hadith_embeddings.progress"
    meta_path = staging_dir / "hadith_embeddings.meta.json"

    ids_hash = hashlib.sha256("\n".join(hadith_ids).encode("utf-8")).hexdigest()
    meta = {"count": n, "dim": dim, "model": _MODEL_NAME, "ids_hash": ids_hash}

    # Reuse an in-progress memmap only when the corpus + model are byte-identical.
    start = 0
    emb = None
    if emb_path.exists() and meta_path.exists() and prog_path.exists():
        try:
            same = _json.loads(meta_path.read_text()) == meta
        except (ValueError, OSError):
            same = False
        if same:
            candidate = np.lib.format.open_memmap(emb_path, mode="r+")
            if tuple(candidate.shape) == (n, dim):
                emb = candidate
                start = int(prog_path.read_text().strip() or "0")
    if emb is None:
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, dim))
        meta_path.write_text(_json.dumps(meta))
        prog_path.write_text("0")
        start = 0

    if start >= n:
        logger.info("dedup_encoding_resume_complete", count=n)
        return emb

    remaining = n - start
    # Only fan out when the process/model-build overhead is amortized by enough
    # remaining work; otherwise the serial path is faster and simpler.
    use_parallel = (
        workers > 1 and model_provider is not None and remaining >= max(2 * batch_size, 2)
    )
    logger.info(
        "dedup_encoding",
        count=n,
        batch_size=batch_size,
        resume_from=start,
        workers=workers if use_parallel else 1,
    )
    if use_parallel:
        assert model_provider is not None  # narrowed by use_parallel
        _encode_range_parallel(
            texts, emb, start, n, batch_size, workers, model_provider, prog_path, mp_start_method
        )
    else:
        _encode_range_serial(texts, emb, start, n, batch_size, prog_path, model)

    emb.flush()
    logger.info("dedup_embeddings_saved", shape=[n, dim], dir=str(staging_dir))
    return emb


def _search_and_collect_resumable(
    *,
    embeddings: npt.NDArray[np.float32],
    hadith_ids: list[str],
    id_to_sect: dict[str, str],
    actual_k: int,
    threshold: float,
    ckpt_dir: Path,
    fingerprint: str,
    cadence: int,
    resume: bool,
    index_available: bool,
    build_index: Callable[[], _Searchable],
    reload_index: Callable[[], _Searchable],
    block_size: int = _DEDUP_SEARCH_BLOCK,
    stop_after: int | None = None,
) -> tuple[list[str], list[str], list[float], list[str], list[bool]]:
    """Search the corpus against a FAISS-like index in resumable row-blocks and
    collect the classified parallel pairs (da#272).

    Split out of :func:`run_dedup` so the crash-resume mechanics are unit-testable
    without faiss / sentence-transformers: the index is supplied via
    ``build_index`` / ``reload_index``, which need only return an object with a
    ``search(matrix, k) -> (scores, indices)`` method.

    On a valid checkpoint (matching layout + fingerprint) AND an available
    persisted index, the accumulators are restored and the scan continues from the
    next block via ``reload_index`` — so the queried index is byte-for-byte the
    crashed run's and an IVF index's random kmeans init cannot diverge the resumed
    search; otherwise it cold-starts via ``build_index``. The result is identical
    to a non-blocked whole-corpus search: each query row is scored independently
    and blocks run in ascending row order, so both the pair set and its emission
    order match a cold run. Returns the five parallel-link column lists.
    """
    n = len(hadith_ids)
    seen_pairs: set[tuple[str, str]] = set()
    ids_a: list[str] = []
    ids_b: list[str] = []
    sim_scores: list[float] = []
    variant_types: list[str] = []
    cross_sects: list[bool] = []
    start_row = 0
    restored = False

    if not resume:
        clear_checkpoint(ckpt_dir)
    else:
        ckpt = load_checkpoint(ckpt_dir)
        if ckpt is not None:
            layout_ok = ckpt.get("schema_version") == _DEDUP_CHECKPOINT_SCHEMA_VERSION
            fp_ok = ckpt.get("fingerprint") == fingerprint
            # Resume only when the persisted index is also present — the loop must
            # query the SAME index the crashed run built (IVF init is random).
            if layout_ok and fp_ok and index_available:
                seen_pairs = {(str(a), str(b)) for a, b in ckpt["seen_pairs"]}
                ids_a = list(ckpt["ids_a"])
                ids_b = list(ckpt["ids_b"])
                sim_scores = list(ckpt["sim_scores"])
                variant_types = list(ckpt["variant_types"])
                cross_sects = list(ckpt["cross_sects"])
                start_row = int(ckpt["processed_rows"])
                restored = True
                log_resume("dedup", skipped=start_row, total=n, pairs=len(ids_a))
            else:
                clear_checkpoint(ckpt_dir)

    index = reload_index() if restored else build_index()

    def _snapshot(processed_rows: int) -> dict[str, object]:
        return {
            "schema_version": _DEDUP_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "processed_rows": processed_rows,
            "seen_pairs": [[a, b] for (a, b) in seen_pairs],
            "ids_a": ids_a,
            "ids_b": ids_b,
            "sim_scores": sim_scores,
            "variant_types": variant_types,
            "cross_sects": cross_sects,
        }

    controller = CheckpointController(cadence, stop_after=stop_after)
    for block_start in range(start_row, n, block_size):
        block_end = min(block_start + block_size, n)
        scores_matrix, indices_matrix = index.search(embeddings[block_start:block_end], actual_k)
        for local_i in range(block_end - block_start):
            i = block_start + local_i
            hid_a = hadith_ids[i]
            for j_idx in range(actual_k):
                neighbor = int(indices_matrix[local_i, j_idx])
                score = float(scores_matrix[local_i, j_idx])

                if neighbor < 0 or neighbor == i:
                    continue
                if score < threshold:
                    continue

                hid_b = hadith_ids[neighbor]

                # Canonical ordering to eliminate symmetric duplicates.
                pair_key = (hid_b, hid_a) if hid_a >= hid_b else (hid_a, hid_b)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                ids_a.append(pair_key[0])
                ids_b.append(pair_key[1])
                sim_scores.append(score)
                variant_types.append(str(_classify_pair(score)))
                cross_sects.append(_is_cross_sect(id_to_sect[pair_key[0]], id_to_sect[pair_key[1]]))

        if controller.batch_complete():
            save_checkpoint(ckpt_dir, _snapshot(block_end))
            logger.info("dedup_checkpoint_saved", processed=block_end, total=n, pairs=len(ids_a))
            if controller.checkpoint_written():
                # --stop-after budget hit (da#276): checkpoint on disk, output not
                # written — perf summary + halt. run_dedup never writes its parquet.
                controller.stop("dedup", processed=block_end, total=n, pairs=len(ids_a))

    return ids_a, ids_b, sim_scores, variant_types, cross_sects


def run_dedup(
    staging_dir: Path,
    *,
    batch_size: int = 256,
    top_k: int = 50,
    threshold: float = 0.70,
    index_type: str = "flat",
    encode_workers: int | None = None,
    resume: bool = True,
    checkpoint_every_n_blocks: int | None = None,
    stop_after: int | None = None,
    require_ml: bool | None = None,
) -> Path:
    """Run full hadith deduplication pipeline.

    Parameters
    ----------
    staging_dir:
        Directory containing hadith Parquet files.
    batch_size:
        Batch size for embedding generation.
    top_k:
        Number of nearest neighbors to retrieve per hadith.
    threshold:
        Minimum cosine similarity to keep a pair (>= 0.70).
    index_type:
        FAISS index type -- ``"flat"`` for IndexFlatIP,
        ``"ivf"`` for IndexIVFFlat (better for large datasets).
    encode_workers:
        Number of processes for the embedding encode (da#246). ``None`` (the
        default) reads ``DEDUP_ENCODE_WORKERS`` from settings; ``0``/unset means
        auto-scale to the box; ``1`` forces the serial fallback. Any value is
        clamped to the physical core count to avoid oversubscription.
    resume:
        When ``True`` (default), a valid crash-resume checkpoint of the FAISS
        search + pair-collection phase (da#272) is restored so the phase
        continues from the last completed block instead of rescanning. ``False``
        discards any checkpoint and re-scans from the top (the embedding memmap
        is still reused — that resume is governed separately, da#245).
    checkpoint_every_n_blocks:
        Persist the pair-collection state every N query blocks. ``None`` reads
        ``DEDUP_CHECKPOINT_EVERY_N_BLOCKS`` then falls back to the default.
    stop_after:
        Bounded partial-run probe (da#276): stop after this many checkpoint writes,
        leaving the checkpoint on disk and WITHOUT writing ``parallel_links.parquet``
        — raises :class:`~src.resolve._checkpoint.StopAfterReached`. ``None`` runs to
        completion.
    require_ml:
        Whether this stage's declared ML dependencies are mandatory (da#309).
        ``None`` (the default) reads ``DEDUP_REQUIRE_ML`` from settings, which
        defaults to ``True``. When ``True`` and a declared dependency is absent,
        :class:`~src.resolve._deps.MissingDependencyError` is raised instead of
        emitting a zero-row output. ``False`` is an explicit opt-in to the
        deterministic no-model fallback (``parallels.py``).

    Returns
    -------
    Path to the output ``parallel_links.parquet`` file. The file is written
    even when zero pairs are found (empty table matching the schema).

    Raises
    ------
    MissingDependencyError
        When the stage is required (see ``require_ml``) but one of its declared
        dependencies is not importable. Previously this returned a zero-row
        ``parallel_links.parquet`` that downstream read as "no parallels found"
        (da#309).
    """
    from src.config import get_settings

    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 0. Declared-dependency guard (da#309).
    #
    # Checked at *dedup's* entry, before dedup reads any input: an enabled stage
    # whose declared dependency is absent is an environment defect regardless of
    # whether its input happens to be empty. `require_ml=False` (DEDUP_REQUIRE_ML)
    # is the one legitimate skip — an explicit opt-in to the deterministic
    # no-model fallback in `parallels.py`, not an inferred one.
    #
    # Note this is dedup's entry, NOT resolve's. dedup is the ninth of the ten
    # `RESOLVE_STEP_ORDER` steps: ner, disambiguate, bio_promote, cluster,
    # narrator_split, reconcile, tabaqa_dates and muhaddithat_links have all
    # already run, and bio_promote has written `narrators_canonical.parquet`.
    # `EXIT_MISSING_DEPENDENCY` therefore means "a stage aborted mid-pipeline with
    # prior artifacts on disk", not "the pipeline refused to start" — an operator
    # seeing exit 4 must not assume nothing was written. `MissingDependencyError`
    # subclasses `BaseException` precisely so it escapes those stages' handlers.
    # ------------------------------------------------------------------
    if require_ml is None:
        require_ml = get_settings().dedup_require_ml
    missing = missing_dependencies(_DECLARED_DEPENDENCIES)
    if missing:
        if require_ml:
            raise MissingDependencyError(
                stage="dedup",
                missing=missing,
                dependency_group="ml",
                remediation="uv sync --group ml",
            )
        logger.warning(
            "dedup_skipped_degraded",
            missing=missing,
            msg=(
                "ML deps absent and DEDUP_REQUIRE_ML=false -- writing an empty "
                "parallel_links.parquet on purpose; the deterministic parallels.py "
                "detector still runs. This is NOT a 'no parallels found' result."
            ),
        )
        # da#378: stamp DEGRADED_NO_ML so this empty artifact is no longer
        # byte-identical to a true negative (semantic=RAN, 0 rows).
        return _write_empty_output(staging_dir, DetectorStatus.DEGRADED_NO_ML)

    # ------------------------------------------------------------------
    # 1. Load hadith texts
    # ------------------------------------------------------------------
    hadith_files = _hadith_files(staging_dir)
    if not hadith_files:
        # Case A: no input files at all. This is an upstream defect (parse never
        # ran / wrong staging_dir), NOT a true negative -- but `run_dedup` is also
        # called directly on empty tmp dirs by unit tests, so making it fatal is a
        # behaviour change tracked separately. Kept distinct from Case B below so
        # the two are no longer structurally indistinguishable.
        logger.warning("dedup_no_hadith_files", staging_dir=str(staging_dir))
        return _write_empty_output(staging_dir, DetectorStatus.NO_INPUT)

    hadith_ids, texts, sects = _load_hadith_texts(staging_dir)
    if not texts:
        # Case B: input files exist, but no row carries a non-empty English matn.
        # A genuinely empty input for an English-matn semantic detector.
        logger.warning("dedup_no_texts", files=len(hadith_files))
        return _write_empty_output(staging_dir, DetectorStatus.NO_TEXTS)

    # ------------------------------------------------------------------
    # 2. Generate embeddings
    # ------------------------------------------------------------------
    import numpy  # noqa: F401  (used by _encode_with_resume; presence proven above)
    from sentence_transformers import SentenceTransformer

    logger.info("dedup_loading_model", model=_MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)

    # Persist the id mapping up front so a resumed encode can validate the corpus.
    with open(staging_dir / "hadith_id_mapping.json", "w") as f:
        json.dump(hadith_ids, f)

    # ------------------------------------------------------------------
    # 3. Generate (or resume) embeddings — disk-backed memmap, crash-resumable.
    # ------------------------------------------------------------------
    # `embeddings` is the memmap itself: already float32 and C-contiguous, so it
    # feeds FAISS directly without an extra full-size in-RAM copy.
    from src.config import get_settings

    workers_setting = (
        encode_workers if encode_workers is not None else get_settings().dedup_encode_workers
    )
    workers = _resolve_encode_workers(workers_setting, len(texts))
    embeddings = _encode_with_resume(
        texts,
        hadith_ids,
        staging_dir,
        model,
        batch_size,
        workers=workers,
        model_provider=_build_default_model,
    )

    # ------------------------------------------------------------------
    # 4. Build FAISS index & search
    # ------------------------------------------------------------------
    import faiss  # presence proven by the entry guard (da#309)

    n = len(hadith_ids)
    dim = embeddings.shape[1]
    index_path = staging_dir / _FAISS_INDEX_FILENAME
    actual_k = min(top_k + 1, n)  # +1 to account for self-match

    # ------------------------------------------------------------------
    # 4. FAISS index + resumable search/pair-collection (da#272). The corpus
    # identity + every param that changes which pairs are emitted go into the
    # fingerprint, so a checkpoint taken against a different corpus/threshold/index
    # is discarded rather than splicing incompatible pairs. The index build/reload
    # is passed in as callables so the resumable loop is testable without faiss.
    # ------------------------------------------------------------------
    ckpt_dir = checkpoint_dir(staging_dir, "dedup")
    fingerprint = hash_strings(
        _id_set_hash(hadith_ids), _MODEL_NAME, index_type, actual_k, round(threshold, 6), n
    )
    cadence = resolve_cadence(
        checkpoint_every_n_blocks,
        "DEDUP_CHECKPOINT_EVERY_N_BLOCKS",
        _DEDUP_CHECKPOINT_EVERY_N_BLOCKS,
    )
    id_to_sect: dict[str, str] = dict(zip(hadith_ids, sects))

    def _build_index() -> faiss_mod.Index:
        if index_type == "ivf":
            nlist = min(100, n)
            quantizer = faiss.IndexFlatIP(dim)
            idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(embeddings)
            idx.nprobe = min(10, nlist)
        else:
            idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
        faiss.write_index(idx, str(index_path))
        logger.info("dedup_index_built", index_type=index_type, vectors=idx.ntotal)
        return idx

    def _reload_index() -> faiss_mod.Index:
        idx = faiss.read_index(str(index_path))
        logger.info("dedup_index_reloaded", index_type=index_type, vectors=idx.ntotal)
        return idx

    ids_a, ids_b, sim_scores, variant_types, cross_sects = _search_and_collect_resumable(
        embeddings=embeddings,
        hadith_ids=hadith_ids,
        id_to_sect=id_to_sect,
        actual_k=actual_k,
        threshold=threshold,
        ckpt_dir=ckpt_dir,
        fingerprint=fingerprint,
        cadence=cadence,
        resume=resume,
        index_available=index_path.exists(),
        build_index=_build_index,
        reload_index=_reload_index,
        stop_after=stop_after,
    )

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    table = pa.table(
        {
            "hadith_id_a": pa.array(ids_a, type=pa.string()),
            "hadith_id_b": pa.array(ids_b, type=pa.string()),
            "similarity_score": pa.array(sim_scores, type=pa.float32()),
            "variant_type": pa.array(variant_types, type=pa.string()),
            "cross_sect": pa.array(cross_sects, type=pa.bool_()),
        },
        schema=PARALLEL_LINKS_SCHEMA,
    )

    # da#378: the semantic detector executed its real algorithm, so this is a
    # true measurement — RAN with len(ids_a) rows (zero rows here IS a true
    # negative, distinct from the DEGRADED_NO_ML empty above). ``deterministic``
    # is NOT_RUN in dedup's own artifact; run_all's compose fills it in.
    output_path = write_parallel_links(
        table,
        staging_dir / "parallel_links.parquet",
        DetectorProvenance(
            semantic=DetectorStatus.RAN,
            semantic_rows=len(ids_a),
            deterministic=DetectorStatus.NOT_RUN,
            deterministic_rows=0,
        ),
    )

    # The search + collection phase completed and its output is on disk — drop the
    # checkpoint so the next cold run doesn't spuriously resume (da#272).
    clear_checkpoint(ckpt_dir)

    # ------------------------------------------------------------------
    # 7. Summary logging
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t0
    tier_counts: dict[str, int] = {vt.value: 0 for vt in VariantType}
    cross_sect_count = 0
    for vt, cs in zip(variant_types, cross_sects):
        tier_counts[vt] = tier_counts.get(vt, 0) + 1
        if cs:
            cross_sect_count += 1

    logger.info(
        "dedup_complete",
        total_pairs=len(ids_a),
        verbatim=tier_counts[VariantType.VERBATIM],
        close_paraphrase=tier_counts[VariantType.CLOSE_PARAPHRASE],
        thematic=tier_counts[VariantType.THEMATIC],
        cross_sect=cross_sect_count,
        elapsed_seconds=round(elapsed, 2),
    )
    return output_path


def _write_empty_output(staging_dir: Path, semantic_status: DetectorStatus) -> Path:
    """Write an empty parallel_links.parquet stamped with ``semantic_status``.

    ``semantic_status`` records *why* the semantic side is empty — the whole
    point of da#378: a ``DEGRADED_NO_ML`` empty (dedup skipped its algorithm) is
    no longer byte-identical to a ``RAN`` zero-row table (dedup found nothing).
    """
    table = PARALLEL_LINKS_SCHEMA.empty_table()
    output_path = write_parallel_links(
        table,
        staging_dir / "parallel_links.parquet",
        DetectorProvenance(
            semantic=semantic_status,
            semantic_rows=0,
            deterministic=DetectorStatus.NOT_RUN,
            deterministic_rows=0,
        ),
    )
    logger.info("dedup_empty_output", path=str(output_path), semantic_status=semantic_status.value)
    return output_path


def run(
    staging_dir: Path, output_dir: Path, *, resume: bool = True, stop_after: int | None = None
) -> list[Path]:
    """Entry point matching the resolve pipeline interface.

    Delegates to ``run_dedup`` and wraps the result in a list for compatibility
    with the resolve orchestrator. ``resume`` (da#272) threads to the FAISS
    search + pair-collection checkpoint; ``False`` forces that phase to re-scan.
    ``stop_after`` (da#276) bounds that phase to N checkpoint writes then raises
    ``StopAfterReached`` (no output written).
    """
    path = run_dedup(staging_dir, resume=resume, stop_after=stop_after)
    if path.exists():
        return [path]
    return []
