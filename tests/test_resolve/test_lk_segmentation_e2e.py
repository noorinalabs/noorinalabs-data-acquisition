"""End-to-end regression for da#146: a voweled lk-corpus chain must resolve to
*multiple* canonical narrators, not a single un-segmented blob.

Flow exercised: lk CSV (fully voweled Arabic isnad)
  → :func:`src.parse.lk_corpus.run`        (narrator_mentions_lk.parquet)
  → :func:`src.resolve.ner.run`            (narrator_mentions_resolved.parquet)
  → :func:`src.resolve.disambiguate.run`   (narrators_canonical.parquet)

Before da#146 the diacritic-free transmission patterns never matched the voweled
chain, so each whole chain became one mention → one "narrator". This test pins
that a single chain now yields several distinct canonical narrators.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.parse.lk_corpus import LK_COLUMNS
from src.parse.lk_corpus import run as lk_run
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.resolve import disambiguate, ner

# Sahih al-Bukhari hadith 1 — five+ named narrators in a fully-voweled isnad.
_BUKHARI_H1_ISNAD_AR = (
    "حَدَّثَنَا الْحُمَيْدِيُّ عَبْدُاللَّهِ بْنُ الزُّبَيْرِ، قَالَ حَدَّثَنَا "
    "سُفْيَانُ، قَالَ حَدَّثَنَا يَحْيَى بْنُ سَعِيدٍ الأَنْصَارِيُّ، قَالَ "
    "أَخْبَرَنِي مُحَمَّدُ بْنُ إِبْرَاهِيمَ التَّيْمِيُّ، أَنَّهُ سَمِعَ "
    "عَلْقَمَةَ بْنَ وَقَّاصٍ اللَّيْثِيَّ"
)


def _write_lk_csv(path: Path) -> None:
    """Write a one-row lk CSV (16-column layout) with a voweled Arabic isnad."""
    row = {c: "" for c in LK_COLUMNS}
    row.update(
        {
            "Chapter_Number": "1",
            "Chapter_English": "Revelation",
            "Chapter_Arabic": "بدء الوحي",
            "Section_Number": "1",
            "Hadith_number": "1",
            "English_Hadith": "Narrated Umar: Actions are by intentions",
            "English_Isnad": "Narrated 'Umar bin Al-Khattab:",
            "English_Matn": "Actions are by intentions",
            "Arabic_Isnad": _BUKHARI_H1_ISNAD_AR,
            "Arabic_Matn": "إنما الأعمال بالنيات",
            "Arabic_Hadith": _BUKHARI_H1_ISNAD_AR,
            "English_Grade": "Sahih",
            "Arabic_Grade": "صحيح",
        }
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def _seed_bio_candidate(staging_dir: Path) -> None:
    """Write a minimal narrators_bio parquet so disambiguate has candidates.

    ``disambiguate.run`` early-returns when no biographical candidates exist;
    one Sufyan bio both unblocks the run and provides an exact-match target for
    the segmented ``سفيان`` mention.
    """
    fields = {f.name: None for f in NARRATOR_BIO_SCHEMA}
    fields.update(
        {
            "bio_id": "bio:test:sufyan",
            "source": "test",
            "name_ar": "سفيان",
            "name_ar_normalized": "سفيان",
            "name_en": "Sufyan",
            "death_year_ah": 198,
        }
    )
    arrays = {f.name: pa.array([fields[f.name]], type=f.type) for f in NARRATOR_BIO_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)
    pq.write_table(table, staging_dir / "narrators_bio_test.parquet")


def test_voweled_lk_chain_yields_multiple_canonical_narrators(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "lk").mkdir(parents=True)
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "output"
    staging_dir.mkdir()
    output_dir.mkdir()

    _write_lk_csv(raw_dir / "lk" / "albukhari.csv")

    # 1) Parse — the lk parser must emit several mention rows for the one chain.
    lk_run(raw_dir, staging_dir)
    mentions = pq.read_table(staging_dir / "narrator_mentions_lk.parquet")
    ar_mentions = [
        mentions.column("name_ar")[i].as_py()
        for i in range(mentions.num_rows)
        if mentions.column("name_ar")[i].as_py()
    ]
    assert len(ar_mentions) >= 4, f"chain not segmented: {ar_mentions}"
    # No mention should be the whole chain (the pre-da#146 blob failure mode).
    assert all(len(m) < len(_BUKHARI_H1_ISNAD_AR) // 2 for m in ar_mentions)

    # 2) NER consolidation.
    ner.run(staging_dir, output_dir)
    resolved = pq.read_table(output_dir / "narrator_mentions_resolved.parquet")
    assert resolved.num_rows >= 4

    # 3) Disambiguate — must materialize multiple canonical narrators, not 0/1.
    _seed_bio_candidate(staging_dir)
    disambiguate.run(staging_dir, output_dir)
    canonical = pq.read_table(output_dir / "narrators_canonical.parquet")
    assert canonical.num_rows >= 4

    # The seeded Sufyan bio should have matched the segmented سفيان mention.
    names_en = {canonical.column("name_en")[i].as_py() for i in range(canonical.num_rows)}
    assert "Sufyan" in names_en
