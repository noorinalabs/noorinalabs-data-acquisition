"""Tests for the da#318 post-hoc hadith-matn markup scrub (``scripts/scrub_matn_markup.py``).

The scrub replays the parser's ``_strip_structural_tags`` transform over the
persisted ``matn_ar`` column of an already-built ``hadiths_sanadset.parquet``, so a
scrubbed row is byte-identical to what the fixed parser would emit. These tests
assert the leak is removed from ``matn_ar``, ``isnad_raw_ar`` is left tag-bearing,
the scrub is idempotent, and the input directory is never mutated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.scrub_matn_markup import HADITHS, run_scrub
from src.parse.schemas import HADITH_SCHEMA

_MARKUP_PRESENT = re.compile(r"<\s*/?\s*(?:NAR|SANAD|MATN)\s*/?\s*>")

# Row 0: a malformed row whose matn leaked structural markup (the doubled-SANAD
# capture) while isnad_raw_ar legitimately carries <NAR> tags.
# Row 1: a clean row — matn markup-free, isnad tag-bearing.
_MATN = [
    "إنما الأعمال بالنيات <SANAD><NAR>علي</NAR></SANAD><MATN>وإنما لكل امرئ ما نوى",
    "لا ضرر ولا ضرار",
]
_ISNAD = [
    "<NAR>محمد بن عبدالله</NAR>",
    "<NAR>أبو هريرة</NAR>",
]


def _write_input(staging_dir: Path) -> None:
    """Build a minimal two-row hadith parquet conforming to HADITH_SCHEMA."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    n = len(_MATN)
    columns: dict[str, list[object]] = {name: [None] * n for name in HADITH_SCHEMA.names}
    columns["source_id"] = [f"sanadset:0:0:{i}" for i in range(n)]
    columns["source_corpus"] = ["sanadset"] * n
    columns["collection_name"] = ["1"] * n
    columns["matn_ar"] = list(_MATN)
    columns["isnad_raw_ar"] = list(_ISNAD)
    columns["full_text_ar"] = list(_MATN)
    columns["sect"] = ["sunni"] * n
    table = pa.table({name: columns[name] for name in HADITH_SCHEMA.names}, schema=HADITH_SCHEMA)
    pq.write_table(table, staging_dir / HADITHS)


def test_scrub_strips_matn_markup_but_keeps_isnad(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "scrubbed"
    _write_input(staging_dir)

    run_scrub(staging_dir, output_dir)

    rows = pq.read_table(output_dir / HADITHS).to_pylist()
    # Malformed row: matn is now free of structural markup, both Arabic segments kept.
    assert _MARKUP_PRESENT.search(rows[0]["matn_ar"]) is None, rows[0]["matn_ar"]
    assert "إنما الأعمال بالنيات" in rows[0]["matn_ar"]
    assert "وإنما لكل امرئ ما نوى" in rows[0]["matn_ar"]
    # isnad_raw_ar is UNTOUCHED — still carries its <NAR> tags for the chain extractor.
    assert rows[0]["isnad_raw_ar"] == _ISNAD[0]
    assert rows[1]["isnad_raw_ar"] == _ISNAD[1]
    # Clean row's matn is unchanged.
    assert rows[1]["matn_ar"] == _MATN[1]


def test_scrub_preserves_schema_and_row_count(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "scrubbed"
    _write_input(staging_dir)

    run_scrub(staging_dir, output_dir)

    table = pq.read_table(output_dir / HADITHS)
    assert table.schema == HADITH_SCHEMA
    assert table.num_rows == len(_MATN)


def test_scrub_is_idempotent(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    first = tmp_path / "scrub1"
    second = tmp_path / "scrub2"
    _write_input(staging_dir)

    run_scrub(staging_dir, first)
    # Re-scrubbing the already-scrubbed output changes nothing further.
    run_scrub(first, second)

    assert pq.read_table(first / HADITHS).to_pydict() == pq.read_table(second / HADITHS).to_pydict()


def test_scrub_does_not_mutate_input(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "scrubbed"
    _write_input(staging_dir)

    run_scrub(staging_dir, output_dir)

    # Input parquet still carries the original leaked markup — read-only over input.
    original = pq.read_table(staging_dir / HADITHS).to_pylist()
    assert _MARKUP_PRESENT.search(original[0]["matn_ar"]) is not None


def test_scrub_rejects_output_equal_to_input(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    _write_input(staging_dir)

    from scripts.scrub_matn_markup import main

    argv = [
        "scrub_matn_markup.py",
        "--staging-dir",
        str(staging_dir),
        "--output-dir",
        str(staging_dir),
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            main()
