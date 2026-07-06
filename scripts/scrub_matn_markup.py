#!/usr/bin/env python3
"""Post-hoc scrub: strip residual structural markup from an already-built hadith parquet.

da#318. Malformed upstream ``sanadset.csv`` rows carry a doubled
``<SANAD>…<MATN>`` segment closed by a single ``</MATN>``, so the parser's
first-open→first-close matn capture (``src.parse.sanadset._MATN_RE``) swallowed a
second ``<MATN>`` open, an orphan ``</SANAD>``, and ``<NAR>…</NAR>`` tags verbatim
into ``matn_ar`` (measured: 87,739 / 650,986 = 13.5% of rows). The parser is now
fixed at source (``_strip_structural_tags``), but the already-built
``hadiths_sanadset.parquet`` under ``data/staging`` still carries the leaked
markup. Rather than pay for a full re-parse, this scrub re-applies the *identical*
strip function to the stored ``matn_ar`` so the artifact matches a fresh parse.

Sibling of :mod:`scripts.scrub_matn_canonical` (da#311, narrator-canonical side)
and :mod:`scripts.scrub_relational_pollution` (da#247). The equivalence argument
is the strongest of the three: this script imports and calls the SAME
``_strip_structural_tags`` the parser applies at store time, so a scrubbed row is
byte-identical to what the fixed parser would have produced — the scrub is a
literal replay of the parse-time transform over the persisted column, not an
approximation of it.

Scope (matches the da#318 AC — tag-strip, NOT re-segmentation):

* **Only ``matn_ar``** is rewritten. ``isnad_raw_ar`` is left untouched — it
  legitimately carries ``<NAR>`` tags that ``_extract_narrator_mentions`` consumes
  (74.5% of rows), so scrubbing it would corrupt the chain input.
* The ~2 narrator mentions/malformed row that the doubled-SANAD segment contributes
  to the chain are NOT recovered here — that is a re-segmentation concern, tracked
  as a da#318 follow-up, out of this scrub's scope.

Idempotent: ``_strip_structural_tags`` is a pure, idempotent function of
``matn_ar`` (its output re-strips to itself), so re-running against an
already-scrubbed output rewrites nothing further. Read-only over the input
directory — never mutates ``data/staging`` in place; always writes to a separate
``--output-dir``, intended for a later targeted reload (an owner-gated data op,
NOT run here).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.sanadset import _strip_structural_tags

HADITHS = "hadiths_sanadset.parquet"


def _scrub_hadiths(staging_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Strip residual structural markup from ``matn_ar`` in the hadith parquet.

    Returns ``(rows_total, rows_rewritten)`` where ``rows_rewritten`` counts rows
    whose stored ``matn_ar`` changed under the strip (i.e. rows that carried leaked
    ``<NAR>``/``<SANAD>``/``<MATN>`` markup). ``isnad_raw_ar`` and every other column
    are passed through unchanged.
    """
    table = pq.read_table(staging_dir / HADITHS)
    matn_ar = table.column("matn_ar").to_pylist()

    cleaned = [_strip_structural_tags(m) for m in matn_ar]
    rows_rewritten = sum(1 for old, new in zip(matn_ar, cleaned, strict=True) if old != new)

    new_matn = pa.array(cleaned, type=pa.string())
    scrubbed = table.set_column(table.schema.get_field_index("matn_ar"), "matn_ar", new_matn)

    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(scrubbed, output_dir / HADITHS)
    return table.num_rows, rows_rewritten


def run_scrub(staging_dir: Path, output_dir: Path) -> None:
    total, rewritten = _scrub_hadiths(staging_dir, output_dir)
    print(
        f"{HADITHS}: {total} rows "
        f"({rewritten} matn_ar rewritten to strip <NAR>/<SANAD>/<MATN> markup; "
        f"isnad_raw_ar untouched)"
    )
    print("scrub complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/staging"),
        help="Input staging directory (read-only; default: data/staging)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the scrubbed artifact (created if absent; "
        "never the same as --staging-dir)",
    )
    args = parser.parse_args()

    if args.output_dir.resolve() == args.staging_dir.resolve():
        parser.error("--output-dir must not be the same as --staging-dir")

    run_scrub(args.staging_dir, args.output_dir)


if __name__ == "__main__":
    main()
