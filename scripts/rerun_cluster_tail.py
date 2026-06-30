"""Re-run the resolve tail (cluster → dates → muhaddithat → dedup → parallels).

#723 helper: the load-bearing NER + disambiguate + bio_promote output already
sits in ``curated/`` from a prior partial run; only the downstream tail needs
re-running (e.g. after the fuzzy_cluster thread-pool perf fix). This mirrors
``src.resolve.run_all`` steps 3.5–6 exactly, against the on-disk ``curated`` +
``staging`` dirs, so it resumes from cluster without redoing the ~3.4h head.

Run from the repo root:  uv run python -m scripts.rerun_cluster_tail
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from src.config import get_settings
from src.resolve import (
    _compose_parallel_links,
    _read_parallel_links,
    date_reconcile,
    dedup,
    fuzzy_cluster,
    muhaddithat_links,
    parallels,
    tabaqa_dates,
)
from src.utils.logging import get_logger

# Skip the fuzzy_cluster recall-increment by default for the main#723 reload:
# its cdist scoring is O(m²) per block and projects to ~14h+ even capped at 250
# (the scored/sec throughput is a fixed cdist-hardware floor, invariant to the
# cap — only candidate VOLUME scales with it). It is non-load-bearing (a recall
# increment on the exact-name pass) and can be re-applied to the graph
# incrementally once a tractable candidate-generation approach (ANN blocking,
# not full O(m²) cdist) is built. Set RERUN_RUN_CLUSTER=1 to force it on.
_RUN_CLUSTER = os.environ.get("RERUN_RUN_CLUSTER") == "1"

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    staging_dir = Path(settings.data_staging_dir)
    output_dir = Path(settings.data_curated_dir)
    canonical_path = output_dir / "narrators_canonical.parquet"
    mentions_path = output_dir / "narrator_mentions_resolved.parquet"

    logger.info("rerun_tail_start", staging_dir=str(staging_dir), output_dir=str(output_dir))

    # Step 3.5: Fuzzy cross-source clustering. SKIPPED by default for #723 — see
    # the _RUN_CLUSTER note above. Non-load-bearing; deferred to a follow-up.
    if _RUN_CLUSTER:
        try:
            logger.info("resolve_step", step="cluster", status="running")
            cluster_metrics = fuzzy_cluster.cluster_canonical_narrators(
                canonical_path,
                mentions_path=mentions_path if mentions_path.exists() else None,
            )
            logger.info(
                "resolve_step",
                step="cluster",
                status="complete",
                merged=cluster_metrics.merged_records,
                clusters=cluster_metrics.multi_member_clusters,
            )
        except Exception:  # noqa: BLE001
            logger.error("resolve_step_failed", step="cluster", traceback=traceback.format_exc())
    else:
        logger.info("resolve_step", step="cluster", status="skipped", reason="deferred_723")

    # Step 3.6: Multi-source date reconciliation.
    try:
        logger.info("resolve_step", step="reconcile_dates", status="running")
        date_reconcile.reconcile_canonical_dates(staging_dir, output_dir)
        logger.info("resolve_step", step="reconcile_dates", status="complete")
    except Exception:  # noqa: BLE001
        logger.error(
            "resolve_step_failed", step="reconcile_dates", traceback=traceback.format_exc()
        )

    # Step 3.65: ṭabaqa → estimated-window fallback.
    try:
        logger.info("resolve_step", step="tabaqa_dates", status="running")
        tabaqa_dates.apply_tabaqa_fallback(output_dir)
        logger.info("resolve_step", step="tabaqa_dates", status="complete")
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="tabaqa_dates", traceback=traceback.format_exc())

    # Step 3.7: Curated muhaddithat orphan mention-links.
    try:
        logger.info("resolve_step", step="muhaddithat_links", status="running")
        muhaddithat_links.build_muhaddithat_mention_links(
            output_dir,
            canonical_path=canonical_path if canonical_path.exists() else None,
        )
        logger.info("resolve_step", step="muhaddithat_links", status="complete")
    except Exception:  # noqa: BLE001
        logger.error(
            "resolve_step_failed", step="muhaddithat_links", traceback=traceback.format_exc()
        )

    # Step 4: Dedup (semantic). Capture its links before parallels overwrites them.
    semantic_links = None
    try:
        logger.info("resolve_step", step="dedup", status="running")
        dedup.run(staging_dir, output_dir)
        semantic_links = _read_parallel_links(staging_dir)
        logger.info("resolve_step", step="dedup", status="complete")
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="dedup", traceback=traceback.format_exc())

    # Step 5: Deterministic lexical parallels.
    deterministic_links = None
    try:
        logger.info("resolve_step", step="parallels", status="running")
        parallels.run(staging_dir, output_dir)
        deterministic_links = _read_parallel_links(staging_dir)
        logger.info("resolve_step", step="parallels", status="complete")
    except Exception:  # noqa: BLE001
        logger.error("resolve_step_failed", step="parallels", traceback=traceback.format_exc())

    # Step 6: Compose both detectors' links into the shared artifact.
    composed = _compose_parallel_links(staging_dir, semantic_links, deterministic_links)
    logger.info("rerun_tail_complete", composed=str(composed) if composed is not None else None)


if __name__ == "__main__":
    main()
