#!/usr/bin/env python3
"""Post-hoc scrub: re-prune FINISHED curated artifacts with the corrected da#311 gate.

da#311. Run 5 (clean full resolve) completed with residual benediction/matn
contamination surviving in ``narrators_canonical.parquet`` (measured with the
*corrected* gate — the un-hardened gate that actually produced run-5 does not
see these rows as matn at all, which is the bug; see the PR body for the exact
before/after row counts and percentages measured against the live data).
Owner-approved: fix the gate (``src/parse/name_quality.py``) AND re-prune the
already-built run-5 artifacts with it rather than pay for a full ~7h resolve
re-run.

This is the sibling of :mod:`scripts.scrub_relational_pollution` (da#247/da#253),
generalized in two ways:

1. It re-checks **every** canonical row against the CURRENT ``clean_narrator_name``
   (whatever classes it rejects today — matn, relational-mubham, English prose,
   …), not a narrower matn-only predicate, so it is a superset re-application of
   the precedent's "curated-scrub == NER-with-filter" equivalence argument.
2. It extends the referential re-prune beyond the two narrator-mention files to
   ``merge_log.parquet`` (drop rows whose ``canonical_id`` no longer survives) and
   ``parallel_links.parquet`` (checked defensively for a canonical-narrator
   reference column; the live schema is hadith-to-hadith only — see the module
   docstring note below — so today this is a pass-through copy, not a filter).

Equivalence argument (mc>1 rows — the precedent's singleton-cluster argument
does not on its own cover these): a dropped canonical row's mentions are matn
text themselves (the "name" IS the hadith body / dialogue fragment), so — like
the da#247 relational-pronoun case — they never legitimately cluster with a
*different* real narrator; dropping the mention is equivalent to that mention
never having survived NER, not a merge that could ever affect a surviving
narrator's own identity. Mentions are DROPPED, never reassigned to a
different canonical id (the source text was never a real name to begin with).

Idempotent: re-running against an already-scrubbed output finds nothing further
to drop (``clean_narrator_name`` is a pure function of ``name_ar_normalized``,
which does not change once scrubbed). Read-only over the input directories —
never mutates ``data/curated`` / ``data/staging`` in place; always writes to a
separate ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.name_quality import clean_narrator_name

NARRATORS = "narrators_canonical.parquet"
MENTIONS = "narrator_mentions_resolved.parquet"
MERGE_LOG = "merge_log.parquet"
PARALLEL_LINKS = "parallel_links.parquet"

# Additional curated file the da#247 precedent also re-pruned. Not in da#311's
# explicit four-file scope, but present in the live curated/ directory; copied
# through unscrubbed (its own canonical_narrator_id refs, if any, are pruned by
# the same rule as MENTIONS so the output directory stays self-consistent).
MENTIONS_MUHADDITHAT = "narrator_mentions_resolved_muhaddithat.parquet"

# Column names that would carry a canonical-narrator reference on a
# hadith-to-hadith link table, checked defensively (da#311): the live
# parallel_links.parquet schema (hadith_id_a, hadith_id_b, similarity_score,
# variant_type, cross_sect) carries NONE of these — it is hadith-level, not
# narrator-level — so today's run is a pass-through copy. Kept as a forward
# guard in case a future schema version adds a narrator-keyed column.
_CANONICAL_REF_COLUMNS = ("canonical_narrator_id", "canonical_id")


def _prune_canonical(curated_dir: Path, output_dir: Path) -> tuple[int, int, set[str]]:
    """Filter narrators_canonical.parquet through the corrected gate.

    Returns ``(rows_before, rows_after, dropped_canonical_ids)``.
    """
    table = pq.read_table(curated_dir / NARRATORS)
    names = table.column("name_ar_normalized").to_pylist()
    canonical_ids = table.column("canonical_id").to_pylist()
    keep_mask = [clean_narrator_name(n) is not None for n in names]
    dropped_ids = {cid for cid, keep in zip(canonical_ids, keep_mask, strict=True) if not keep}
    kept = table.filter(pa.array(keep_mask))
    pq.write_table(kept, output_dir / NARRATORS)
    return table.num_rows, kept.num_rows, dropped_ids


def _prune_by_canonical_column(
    path: Path,
    output_path: Path,
    column: str,
    dropped_ids: set[str],
) -> tuple[int, int] | None:
    """Drop rows of *path* whose *column* references a dropped canonical id.

    Returns ``(rows_before, rows_after)``, or ``None`` if the file does not
    exist (some curated directories omit optional files — mirrors the
    da#247 precedent's ``if not path.exists(): continue`` behavior).
    """
    if not path.exists():
        return None
    table = pq.read_table(path)
    if column not in table.schema.names:
        # No canonical reference in this schema — pass through unchanged
        # rather than silently dropping nothing-that-was-checked.
        pq.write_table(table, output_path)
        return table.num_rows, table.num_rows
    refs = table.column(column).to_pylist()
    keep_mask = [ref not in dropped_ids for ref in refs]
    kept = table.filter(pa.array(keep_mask))
    pq.write_table(kept, output_path)
    return table.num_rows, kept.num_rows


def _prune_parallel_links(
    staging_dir: Path, output_dir: Path, dropped_ids: set[str]
) -> tuple[int, int, bool]:
    """Re-prune parallel_links.parquet if (and only if) it carries a canonical ref.

    Returns ``(rows_before, rows_after, had_canonical_column)``.
    """
    path = staging_dir / PARALLEL_LINKS
    output_path = output_dir / PARALLEL_LINKS
    table = pq.read_table(path)
    ref_column = next((c for c in _CANONICAL_REF_COLUMNS if c in table.schema.names), None)
    if ref_column is None:
        pq.write_table(table, output_path)
        return table.num_rows, table.num_rows, False
    refs = table.column(ref_column).to_pylist()
    keep_mask = [ref not in dropped_ids for ref in refs]
    kept = table.filter(pa.array(keep_mask))
    pq.write_table(kept, output_path)
    return table.num_rows, kept.num_rows, True


def run_scrub(curated_dir: Path, staging_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_before, nc_after, dropped_ids = _prune_canonical(curated_dir, output_dir)
    print(
        f"{NARRATORS}: {nc_before} -> {nc_after} "
        f"(dropped {len(dropped_ids)} matn/pollution canonical rows)"
    )

    mentions_result = _prune_by_canonical_column(
        curated_dir / MENTIONS,
        output_dir / MENTIONS,
        "canonical_narrator_id",
        dropped_ids,
    )
    if mentions_result is not None:
        before, after = mentions_result
        print(f"{MENTIONS}: {before} -> {after} (dropped {before - after} mentions)")

    muhaddithat_result = _prune_by_canonical_column(
        curated_dir / MENTIONS_MUHADDITHAT,
        output_dir / MENTIONS_MUHADDITHAT,
        "canonical_narrator_id",
        dropped_ids,
    )
    if muhaddithat_result is not None:
        before, after = muhaddithat_result
        print(f"{MENTIONS_MUHADDITHAT}: {before} -> {after} (dropped {before - after} mentions)")

    merge_log_result = _prune_by_canonical_column(
        curated_dir / MERGE_LOG,
        output_dir / MERGE_LOG,
        "canonical_id",
        dropped_ids,
    )
    if merge_log_result is not None:
        before, after = merge_log_result
        print(f"{MERGE_LOG}: {before} -> {after} (dropped {before - after} rows)")

    pl_before, pl_after, had_ref = _prune_parallel_links(staging_dir, output_dir, dropped_ids)
    note = "" if had_ref else " (no canonical-narrator column in this schema — pass-through)"
    print(f"{PARALLEL_LINKS}: {pl_before} -> {pl_after}{note}")

    print("scrub complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=Path("data/curated"),
        help="Input curated directory (read-only; default: data/curated)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/staging"),
        help="Input staging directory for parallel_links.parquet (read-only; "
        "default: data/staging)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the scrubbed artifacts (created if absent; "
        "never the same as --curated-dir)",
    )
    args = parser.parse_args()

    if args.output_dir.resolve() == args.curated_dir.resolve():
        parser.error("--output-dir must not be the same as --curated-dir")

    run_scrub(args.curated_dir, args.staging_dir, args.output_dir)


if __name__ == "__main__":
    main()
