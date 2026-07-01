"""Hadith deduplication and parallel detection.

Generates sentence-transformer embeddings for hadith matn texts, builds a FAISS
index, and identifies parallel hadith pairs across collections and sects.
"""

from __future__ import annotations

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
from src.resolve.schemas import PARALLEL_LINKS_SCHEMA
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import faiss as faiss_mod
    import numpy as np
    import numpy.typing as npt

logger = get_logger(__name__)

__all__ = ["run", "run_dedup"]

# Source corpora classified by sect
_SUNNI_SOURCES: frozenset[str] = frozenset({"lk", "sanadset", "sunnah", "fawaz", "open_hadith"})
_SHIA_SOURCES: frozenset[str] = frozenset({"thaqalayn"})

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


def _classify_pair(score: float) -> VariantType:
    """Classify a similarity score into a variant type tier."""
    if score >= 0.90:
        return VariantType.VERBATIM
    if score >= 0.80:
        return VariantType.CLOSE_PARAPHRASE
    return VariantType.THEMATIC


def _is_cross_sect(corpus_a: str, corpus_b: str) -> bool:
    """Return True when hadiths come from different sectarian traditions."""
    a_sunni = corpus_a in _SUNNI_SOURCES
    b_sunni = corpus_b in _SUNNI_SOURCES
    a_shia = corpus_a in _SHIA_SOURCES
    b_shia = corpus_b in _SHIA_SOURCES
    return (a_sunni and b_shia) or (a_shia and b_sunni)


def _load_hadith_texts(
    staging_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Load hadith IDs, English matn texts, and source corpora from staging Parquets.

    Returns (hadith_ids, texts, corpora) with null/empty matn_en rows excluded.
    """
    hadith_files = sorted(staging_dir.glob("**/hadiths_*.parquet"))
    if not hadith_files:
        logger.warning("dedup_no_hadith_files", staging_dir=str(staging_dir))
        return [], [], []

    ids: list[str] = []
    texts: list[str] = []
    corpora: list[str] = []
    skipped = 0
    for fpath in hadith_files:
        table = pq.read_table(fpath, columns=["source_id", "matn_en", "source_corpus"])
        for i in range(table.num_rows):
            matn = table.column("matn_en")[i].as_py()
            if not matn or not matn.strip():
                skipped += 1
                continue
            ids.append(table.column("source_id")[i].as_py())
            texts.append(matn)
            corpora.append(table.column("source_corpus")[i].as_py())

    logger.info(
        "dedup_loaded_hadiths",
        included=len(ids),
        skipped=skipped,
        files=len(hadith_files),
    )
    return ids, texts, corpora


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


def run_dedup(
    staging_dir: Path,
    *,
    batch_size: int = 256,
    top_k: int = 50,
    threshold: float = 0.70,
    index_type: str = "flat",
    encode_workers: int | None = None,
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

    Returns
    -------
    Path to the output ``parallel_links.parquet`` file. The file is written
    even when zero pairs are found (empty table matching the schema).
    """
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load hadith texts
    # ------------------------------------------------------------------
    hadith_ids, texts, corpora = _load_hadith_texts(staging_dir)
    if not texts:
        logger.warning("dedup_no_texts")
        return _write_empty_output(staging_dir)

    # ------------------------------------------------------------------
    # 2. Generate embeddings
    # ------------------------------------------------------------------
    try:
        import numpy  # noqa: F401  (availability guard — used by _encode_with_resume)
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "dedup_missing_deps",
            msg="sentence-transformers or numpy not installed -- skipping dedup",
        )
        return _write_empty_output(staging_dir)

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
    try:
        import faiss
    except ImportError:
        logger.error(
            "dedup_missing_faiss",
            msg="faiss-cpu not installed -- skipping similarity search",
        )
        return _write_empty_output(staging_dir)

    dim = embeddings.shape[1]
    faiss_index: faiss_mod.Index
    if index_type == "ivf":
        nlist = min(100, len(texts))
        quantizer = faiss.IndexFlatIP(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss_index.train(embeddings)
        faiss_index.nprobe = min(10, nlist)
    else:
        faiss_index = faiss.IndexFlatIP(dim)

    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(staging_dir / "hadith_embeddings.faiss"))
    logger.info("dedup_index_built", index_type=index_type, vectors=faiss_index.ntotal)

    # Query in one call -- scores shape (n, top_k)
    actual_k = min(top_k + 1, len(texts))  # +1 to account for self-match
    scores_matrix, indices_matrix = faiss_index.search(embeddings, actual_k)

    # ------------------------------------------------------------------
    # 5. Collect and classify pairs
    # ------------------------------------------------------------------
    id_to_corpus: dict[str, str] = dict(zip(hadith_ids, corpora))
    seen_pairs: set[tuple[str, str]] = set()

    ids_a: list[str] = []
    ids_b: list[str] = []
    sim_scores: list[float] = []
    variant_types: list[str] = []
    cross_sects: list[bool] = []

    for i in range(len(hadith_ids)):
        hid_a = hadith_ids[i]
        for j_idx in range(actual_k):
            neighbor = int(indices_matrix[i, j_idx])
            score = float(scores_matrix[i, j_idx])

            if neighbor < 0 or neighbor == i:
                continue
            if score < threshold:
                continue

            hid_b = hadith_ids[neighbor]

            # Canonical ordering to eliminate symmetric duplicates
            if hid_a >= hid_b:
                pair_key = (hid_b, hid_a)
            else:
                pair_key = (hid_a, hid_b)

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            ids_a.append(pair_key[0])
            ids_b.append(pair_key[1])
            sim_scores.append(score)
            variant_types.append(str(_classify_pair(score)))
            cross_sects.append(_is_cross_sect(id_to_corpus[pair_key[0]], id_to_corpus[pair_key[1]]))

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

    output_path = staging_dir / "parallel_links.parquet"
    pq.write_table(table, output_path)

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


def _write_empty_output(staging_dir: Path) -> Path:
    """Write an empty parallel_links.parquet and return its path."""
    table = PARALLEL_LINKS_SCHEMA.empty_table()
    output_path = staging_dir / "parallel_links.parquet"
    pq.write_table(table, output_path)
    logger.info("dedup_empty_output", path=str(output_path))
    return output_path


def run(staging_dir: Path, output_dir: Path) -> list[Path]:
    """Entry point matching the resolve pipeline interface.

    Delegates to ``run_dedup`` and wraps the result in a list for compatibility
    with the resolve orchestrator.
    """
    path = run_dedup(staging_dir)
    if path.exists():
        return [path]
    return []
