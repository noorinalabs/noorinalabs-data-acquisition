"""Tests for the da#366 recovery-measurement harness (``scripts/da366_measure_recovery.py``).

The harness's full run needs the real corpus (data-gated), but its pure
summarisers are exercised here on synthetic rows so the measurement logic itself
is covered — a report that miscounts is worse than no report.
"""

from __future__ import annotations

from scripts.da366_measure_recovery import boundary_precision, summarize_recovery


def test_summarize_recovery_counts_by_convention() -> None:
    texts = [
        # both (opener + >=2 عن)
        "حدثنا مالك عن نافع عن ابن عمر أن النبي قال شيء",
        # an_chain (bare name + >=2 عن)
        "فلان بن فلان عن أبيه عن جده قال شيء",
        # negative: pure matn
        "إنما الأعمال بالنيات",
        # negative: single عن prose
        "باب ما جاء عن النبي",
    ]
    stats = summarize_recovery(texts)
    assert stats.total == 4
    assert stats.recovered == 2
    assert stats.by_convention == {"both": 1, "an_chain": 1}
    assert stats.rate == 0.5


def test_summarize_recovery_empty() -> None:
    stats = summarize_recovery([])
    assert stats.total == 0
    assert stats.recovered == 0
    assert stats.rate == 0.0


def test_boundary_precision_all_clean_when_boundary_faithful() -> None:
    pairs = [
        ("حدثنا مالك عن نافع عن ابن عمر", "أن رسول الله صلى الله عليه وسلم نهى عن بيع الحصاة"),
        ("علي بن إبراهيم عن أبيه عن حماد", "قال الصلاة عماد الدين"),
    ]
    clean, recovered = boundary_precision(pairs)
    assert recovered == 2
    assert clean == 2  # no matn narrator leaked into either recovered isnad
