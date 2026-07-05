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

Beyond dropping, the scrub also **rewrites** a surviving row's **``name_ar``** (the
load-bearing display field the graph loader keys the node name on) to the gate's
recovered value when a real leading narrator is recovered from a polluted span
(``"عاءشه، زوج النبي … قالت"`` → ``"عاءشه"``), and re-derives ``name_ar_normalized``
= ``normalize_arabic(cleaned)`` so the two stay in sync — the "prefer clean over
drop" half, so a recoverable real narrator (and its mention edges) is kept, not
lost. (A first pass cleaned only ``name_ar_normalized`` and left the matn in
``name_ar``, which the loader reads first — this pass fixes the field that
matters.)

Idempotent: :func:`clean_narrator_name_display` is a pure, idempotent function of
``name_ar`` (its output re-cleans to itself), so re-running against an
already-scrubbed output drops nothing further and rewrites nothing further.
Read-only over the input directories — never mutates ``data/curated`` /
``data/staging`` in place; always writes to a separate ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.name_quality import clean_narrator_name_display
from src.utils.arabic import normalize_arabic

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


def _prune_canonical(curated_dir: Path, output_dir: Path) -> tuple[int, int, int, set[str]]:
    """Prune AND clean narrators_canonical.parquet through the corrected gate.

    Cleans the **load-bearing ``name_ar`` field** (:func:`clean_narrator_name_display`)
    — the graph loader keys the Narrator node name on ``name_ar`` first, falling
    back to ``name_ar_normalized`` only when ``name_ar`` is empty, and ``name_ar``
    frequently carries the FULL matn body while ``name_ar_normalized`` was already
    truncated at NER time. A first version that cleaned only ``name_ar_normalized``
    therefore left the matn in the field the graph actually reads.

    Two operations, mirroring the gate's own return contract:

    * **DROP** a row whose cleaner returns ``None`` (unrecoverable matn — a whole
      hadith body, a bare Prophet action, an "asked" dialogue opener, or a
      degenerate one-letter residue).
    * **CLEAN (rewrite)** a surviving row's ``name_ar`` to the recovered value
      (``"عاءشه، زوج النبي … قالت"`` → ``"عاءشه"``, diacritics preserved where the
      recovery is a voweled sub-span), and re-derive ``name_ar_normalized`` =
      ``normalize_arabic(cleaned)`` so the two stay in sync. ``canonical_id`` is
      unchanged, so no mention / merge_log / parallel_links reference is orphaned.
      This is the "prefer clean over drop" half — dropping a recoverable real
      narrator (and its mention edges) would be a regression. Re-clustering the
      rewritten names is a full-resolve concern, out of scope for a post-hoc scrub.

    ``name_en`` is left as stored (sourced-or-empty provenance): the loader
    re-derives the display English from the cleaned ``name_ar`` via
    ``transliterate`` at load time whenever the sourced value is empty, so cleaning
    ``name_ar`` is sufficient for the derived-English case.

    Returns ``(rows_before, rows_after, rows_cleaned, dropped_canonical_ids)`` where
    ``rows_cleaned`` counts surviving rows whose stored ``name_ar`` was rewritten.
    """
    table = pq.read_table(curated_dir / NARRATORS)
    name_ar = table.column("name_ar").to_pylist()
    name_ar_norm = table.column("name_ar_normalized").to_pylist()
    canonical_ids = table.column("canonical_id").to_pylist()

    # Effective display source = name_ar (load-bearing); fall back to the
    # normalized field only when name_ar is empty (mirrors the loader).
    cleaned_display: list[str | None] = []
    for ar, norm in zip(name_ar, name_ar_norm, strict=True):
        source = ar if (ar and ar.strip()) else norm
        cleaned_display.append(clean_narrator_name_display(source))

    keep_mask = [c is not None for c in cleaned_display]
    dropped_ids = {cid for cid, keep in zip(canonical_ids, keep_mask, strict=True) if not keep}
    rows_cleaned = sum(
        1
        for ar, c in zip(name_ar, cleaned_display, strict=True)
        if c is not None and c != (ar or "")
    )

    kept = table.filter(pa.array(keep_mask))
    survivors = [c for c in cleaned_display if c is not None]
    new_ar = pa.array(survivors, type=pa.string())
    new_norm = pa.array([normalize_arabic(c) for c in survivors], type=pa.string())
    kept = kept.set_column(kept.schema.get_field_index("name_ar"), "name_ar", new_ar)
    kept = kept.set_column(
        kept.schema.get_field_index("name_ar_normalized"), "name_ar_normalized", new_norm
    )
    pq.write_table(kept, output_dir / NARRATORS)
    return table.num_rows, kept.num_rows, rows_cleaned, dropped_ids


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

    nc_before, nc_after, nc_cleaned, dropped_ids = _prune_canonical(curated_dir, output_dir)
    print(
        f"{NARRATORS}: {nc_before} -> {nc_after} "
        f"(dropped {len(dropped_ids)} matn/pollution canonical rows; "
        f"cleaned/rewrote {nc_cleaned} recovered names)"
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
