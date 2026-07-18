"""Measure da#366 matn-embedded isnad recovery against the real sanadset corpus.

This is the instrument the issue asks for: run the two-convention splitter over
the real ``No SANAD`` rows and report how many carry a recoverable isnad
(expected ~122,000, a lower bound), broken down by convention; and check the
boundary does not pull matn into the isnad.

**Why it is data-gated.** No real corpus ships in the repo (``data/raw`` is
empty), so this cannot run in CI. The recovery magnitude is a re-run
measurement, not a unit assertion — the splitter's CORRECTNESS is unit-proven in
``tests/test_parse/test_isnad_matn_split.py``; this script produces the MAGNITUDE
once the raw CSVs are present. Run with no data and it prints a loud SKIP and
exits 0 (non-gating), rather than emit a zero that looks like a measurement
("silent zero is not a measurement").

The pure summarisers (:func:`summarize_recovery`, :func:`boundary_precision`)
are unit-tested in ``tests/test_scripts/test_da366_measure.py``.

Usage:
    PYTHONPATH=. python3 scripts/da366_measure_recovery.py [RAW_DIR]
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.parse.isnad_matn_split import split_isnad_matn
from src.parse.narrator_extraction import extract_narrator_mentions


@dataclass(frozen=True)
class RecoveryStats:
    """Recovery tally over a set of candidate matn texts."""

    total: int
    recovered: int
    by_convention: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.recovered / self.total if self.total else 0.0


def summarize_recovery(texts: Iterable[str]) -> RecoveryStats:
    """Run the splitter over *texts* and tally recoveries by convention."""
    total = 0
    conv: Counter[str] = Counter()
    for text in texts:
        total += 1
        result = split_isnad_matn(text)
        if result is not None:
            conv[result.convention] += 1
    return RecoveryStats(total=total, recovered=sum(conv.values()), by_convention=dict(conv))


def boundary_precision(pairs: Iterable[tuple[str, str]]) -> tuple[int, int]:
    """Boundary-precision check over known ``(isnad, matn)`` pairs.

    Reconstructs the untagged shape ``isnad + " " + matn`` and asserts the
    splitter does not pull a matn token into the recovered isnad. Returns
    ``(clean, recovered)`` — ``clean`` is the number of recoveries whose isnad
    narrator set is a subset of the isnad-only segmentation (no matn leaked in).

    Recall is deliberately NOT reported: the tagged rows strip their opener, so
    they are a different textual representation and cannot calibrate the
    splitter's sensitivity (da#366). This measures precision only.
    """
    clean = 0
    recovered = 0
    for isnad, matn in pairs:
        result = split_isnad_matn(f"{isnad} {matn}")
        if result is None:
            continue
        recovered += 1
        truth = {s.name for s in extract_narrator_mentions(isnad, "ar")}
        got = {s.name for s in result.spans}
        if got <= truth:
            clean += 1
    return clean, recovered


def _iter_no_sanad_matns(raw_dir: Path) -> list[str]:  # pragma: no cover - needs real data
    """Yield the matn text of every ``No SANAD`` row across the raw CSVs.

    Mirrors the parser's tag extraction (``src/parse/sanadset.py``) so the
    population matches what the loader would see.
    """
    from src.parse.base import read_csv_robust
    from src.parse.sanadset import _MATN_RE, _SANAD_RE, _extract_tag, _strip_structural_tags

    texts: list[str] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        table, _enc = read_csv_robust(csv_file)
        for i in range(table.num_rows):
            full = table.column("Hadith")[i].as_py() if "Hadith" in table.column_names else None
            if not full:
                continue
            sanad = _extract_tag(_SANAD_RE, full)
            if sanad and sanad.lower() != "no sanad":
                continue
            matn = _strip_structural_tags(_extract_tag(_MATN_RE, full))
            if matn:
                texts.append(matn)
    return texts


def main(argv: list[str]) -> int:  # pragma: no cover - CLI wrapper
    raw_dir = Path(argv[1]) if len(argv) > 1 else Path("data/raw/sanadset")
    if not raw_dir.exists() or not any(raw_dir.glob("*.csv")):
        print(
            f"SKIP: no sanadset CSVs under {raw_dir} — recovery magnitude is a re-run "
            "measurement, not obtainable without the corpus. Splitter correctness is "
            "covered by tests/test_parse/test_isnad_matn_split.py.",
            file=sys.stderr,
        )
        return 0

    stats = summarize_recovery(_iter_no_sanad_matns(raw_dir))
    print(f"No SANAD rows scanned : {stats.total}")
    print(f"isnad recovered       : {stats.recovered} ({stats.rate:.1%})")
    for conv, n in sorted(stats.by_convention.items()):
        print(f"  {conv:9s}: {n}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
