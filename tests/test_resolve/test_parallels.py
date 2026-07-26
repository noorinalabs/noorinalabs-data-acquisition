"""Unit tests for the deterministic lexical PARALLEL_OF detector (da#100)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import structlog

from src.models.enums import VariantType
from src.resolve.parallels import (
    _build_postings,
    _candidate_partners,
    _classify_pair,
    _jaccard,
    _tokenize,
    detect_parallels,
)
from src.utils.arabic import normalize_arabic, normalize_arabic_uncached
from tests.test_graph.conftest import write_hadiths


class TestClassifyPair:
    def test_verbatim(self) -> None:
        assert _classify_pair(0.95) == VariantType.VERBATIM
        assert _classify_pair(0.90) == VariantType.VERBATIM

    def test_close_paraphrase(self) -> None:
        assert _classify_pair(0.85) == VariantType.CLOSE_PARAPHRASE
        assert _classify_pair(0.80) == VariantType.CLOSE_PARAPHRASE

    def test_thematic(self) -> None:
        assert _classify_pair(0.79) == VariantType.THEMATIC
        assert _classify_pair(0.50) == VariantType.THEMATIC


class TestJaccard:
    def test_identical(self) -> None:
        s = frozenset({"a", "b", "c"})
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_partial(self) -> None:
        # {a,b,c} vs {b,c,d}: intersection 2, union 4 -> 0.5
        assert _jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == 0.5

    def test_empty(self) -> None:
        assert _jaccard(frozenset(), frozenset({"a"})) == 0.0


class TestTokenize:
    def test_arabic_preferred_and_normalized(self) -> None:
        # Diacritics differ but normalize_arabic collapses them -> same tokens.
        a = _tokenize("إنَّمَا الأعمال بالنيات", None)
        b = _tokenize("إنما الاعمال بالنيات", None)
        assert a == b
        assert a  # non-empty

    def test_english_fallback_when_no_arabic(self) -> None:
        assert _tokenize(None, "Actions Are By Intentions") == frozenset(
            {"actions", "are", "by", "intentions"}
        )

    def test_empty_when_neither(self) -> None:
        assert _tokenize(None, None) == frozenset()
        assert _tokenize("", "   ") == frozenset()

    def test_matn_body_bypasses_the_normalize_cache(self) -> None:
        # da#495 review: full matn bodies are large and mostly unique, so
        # _tokenize MUST normalize them via the uncached core — feeding them
        # through the memoized normalize_arabic would retain every distinct
        # hadith body for the process lifetime (OOM risk; da#337/#723). Guard
        # the invariant structurally so it can't silently regress.
        assert not hasattr(normalize_arabic_uncached, "cache_info")
        normalize_arabic.cache_clear()
        # A body-sized, unique-per-call Arabic matn.
        body = " ".join(["حدثنا محمد بن إسماعيل قال حدثنا عبد الله"] * 40)
        before = normalize_arabic.cache_info().currsize
        assert _tokenize(body, None)  # non-empty token set
        after = normalize_arabic.cache_info().currsize
        assert after == before == 0, (
            "matn body leaked into the normalize_arabic cache — the parallels "
            "full-corpus path must use normalize_arabic_uncached"
        )


def _coords(**over: object) -> dict[str, object]:
    base: dict[str, object] = {"collection_name": "bukhari", "source_corpus": "sunnah"}
    base.update(over)
    return base


class TestDetectParallels:
    def test_writes_empty_when_no_pairs(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {"source_id": "sunnah:bukhari:1", "matn_en": "the sky is blue", **_coords()},
                {"source_id": "sunnah:bukhari:2", "matn_en": "fish swim in water", **_coords()},
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        assert out.exists()
        assert pq.read_table(out).num_rows == 0

    def test_intra_sect_pair_detected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "actions are judged by their intentions",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "lk:bukhari:1",
                    "matn_en": "actions are judged by the intentions",
                    "sect": "sunni",
                    "source_corpus": "lk",
                    "collection_name": "bukhari",
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        assert len(rows) == 1
        assert rows[0]["cross_sect"] is False
        # Canonical ordering: lk: < sunnah:
        assert rows[0]["hadith_id_a"] == "lk:bukhari:1"
        assert rows[0]["hadith_id_b"] == "sunnah:bukhari:1"

    def test_cross_sect_pair_flagged(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "purification is half of faith",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "thaqalayn:al-kafi:1",
                    "matn_en": "purification is half of the faith",
                    "sect": "shia",
                    "source_corpus": "thaqalayn",
                    "collection_name": "al-kafi",
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        assert len(rows) == 1
        assert rows[0]["cross_sect"] is True

    def test_symmetric_dedup_single_row(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "a b c d e",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "sunnah:bukhari:2",
                    "matn_en": "a b c d e",
                    "sect": "sunni",
                    **_coords(),
                },
            ],
            suffix="a",
        )
        out = detect_parallels(staging, threshold=0.5)
        rows = pq.read_table(out).to_pylist()
        # One identical pair -> exactly one (canonically ordered) row, score 1.0.
        assert len(rows) == 1
        assert rows[0]["similarity_score"] == pytest.approx(1.0)
        assert rows[0]["variant_type"] == VariantType.VERBATIM.value

    def test_threshold_filters_low_similarity(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {
                    "source_id": "sunnah:bukhari:1",
                    "matn_en": "alpha beta gamma delta",
                    "sect": "sunni",
                    **_coords(),
                },
                {
                    "source_id": "sunnah:bukhari:2",
                    "matn_en": "alpha epsilon zeta eta",
                    "sect": "sunni",
                    **_coords(),
                },
            ],
            suffix="a",
        )
        # Jaccard here = 1/7 ~= 0.14 -> below 0.5, no rows.
        assert pq.read_table(detect_parallels(staging, threshold=0.5)).num_rows == 0
        # Lower threshold catches the thematic link.
        rows = pq.read_table(detect_parallels(staging, threshold=0.1)).to_pylist()
        assert len(rows) == 1
        assert rows[0]["variant_type"] == VariantType.THEMATIC.value


class TestBlocking:
    """Inverted-index blocking: the candidate-generation scale fix (da#160)."""

    def test_postings_maps_token_to_hadith_indices(self) -> None:
        indexed = [
            ("a", "sunni", frozenset({"x", "y"})),
            ("b", "sunni", frozenset({"y", "z"})),
        ]
        postings = _build_postings(indexed)
        assert postings["x"] == [0]
        assert postings["y"] == [0, 1]
        assert postings["z"] == [1]

    def test_common_token_skipped_as_blocking_key(self) -> None:
        # "the" appears in every hadith (df 3); "rare" only in 0 and 2 (df 2).
        indexed = [
            ("h0", "sunni", frozenset({"the", "rare"})),
            ("h1", "sunni", frozenset({"the", "other"})),
            ("h2", "sunni", frozenset({"the", "rare"})),
        ]
        postings = _build_postings(indexed)
        # df cap 2: "the" (df 3) is NOT a usable key, so h0's only partner is h2
        # (shared via "rare"), NOT h1 (shared only via the skipped "the").
        partners = _candidate_partners(0, indexed[0][2], postings, max_block_df=2)
        assert partners == {2}
        # Raising the cap above df("the") re-includes h1 as a candidate.
        partners_all = _candidate_partners(0, indexed[0][2], postings, max_block_df=10)
        assert partners_all == {1, 2}


def _pair_set(path: Path) -> set[tuple[str, str]]:
    rows = pq.read_table(path).to_pylist()
    return {(r["hadith_id_a"], r["hadith_id_b"]) for r in rows}


class TestProductionShapedCorpus:
    """End-to-end on a production-shaped corpus: many hadiths, duplicate clusters
    among unique distractors. Proves PARALLEL_OF links are EMITTED at scale and
    that blocking is equivalent to the exhaustive scan it replaced (da#160)."""

    # A 2-token scaffold shared by EVERY hadith — these become high-document-
    # frequency tokens that blocking must skip without losing intra-cluster pairs.
    _SCAFFOLD = "narrated reported"

    def _build_corpus(
        self, staging: Path, *, clusters: int, per_cluster: int, distractors: int
    ) -> set[tuple[str, str]]:
        """Write a hadiths parquet: ``clusters`` groups of ``per_cluster`` identical
        hadiths (each group with its own distinctive vocabulary) plus ``distractors``
        lone hadiths. Returns the set of expected intra-cluster (a,b) pairs."""
        rows: list[dict[str, object]] = []
        expected: set[tuple[str, str]] = set()
        for c in range(clusters):
            # Distinctive, cluster-local vocabulary (df == per_cluster, well under
            # the blocking cap) + the shared high-df scaffold.
            body = " ".join(f"clusterword{c}token{t}" for t in range(8))
            matn = f"{self._SCAFFOLD} {body}"
            ids = []
            for k in range(per_cluster):
                sid = f"sunnah:c{c}:h{k}"
                ids.append(sid)
                rows.append({"source_id": sid, "matn_en": matn, "sect": "sunni"})
            ids.sort()
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    expected.add((ids[a], ids[b]))
        for d in range(distractors):
            body = " ".join(f"distinct{d}word{t}" for t in range(8))
            rows.append(
                {
                    "source_id": f"sunnah:d:{d}",
                    "matn_en": f"{self._SCAFFOLD} {body}",
                    "sect": "sunni",
                }
            )
        write_hadiths(staging, rows, suffix="prod")
        return expected

    def test_emits_edges_and_matches_exhaustive_scan(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        expected = self._build_corpus(staging, clusters=6, per_cluster=3, distractors=60)

        # Blocked run with a small cap (< corpus size): the scaffold tokens
        # ("narrated"/"reported", df = every hadith) are skipped, so blocking does
        # NOT degenerate to the O(n²) scan, yet still recovers the intra-cluster
        # pairs (which share cluster-local tokens with df == per_cluster == 3).
        blocked = detect_parallels(staging, threshold=0.5, max_block_df=10)
        blocked_pairs = _pair_set(blocked)

        # Production-shaped corpus actually produces edges (the da#160 regression
        # was zero) — one row per intra-cluster pair: C(3,2) * 6 = 18.
        assert blocked_pairs == expected
        assert len(blocked_pairs) == 18
        rows = pq.read_table(blocked).to_pylist()
        assert all(r["variant_type"] == VariantType.VERBATIM.value for r in rows)
        assert all(not r["cross_sect"] for r in rows)

        # Exhaustive reference (cap above corpus size ⇒ every token is a key ⇒ the
        # old all-pairs scan). Blocking must yield the identical pair set.
        exhaustive = detect_parallels(staging, threshold=0.5, max_block_df=100_000)
        assert _pair_set(exhaustive) == blocked_pairs

    def test_default_cap_equivalent_to_exhaustive_on_slice(self, tmp_path: Path) -> None:
        # On a loaded slice no larger than the default cap every token's df stays
        # under the cap, so the default-config run must recover the SAME pair set
        # as the exhaustive (cap-above-corpus) scan (loaded-slice / CI guarantee).
        staging = tmp_path / "staging"
        staging.mkdir()
        expected = self._build_corpus(staging, clusters=4, per_cluster=2, distractors=20)

        # Capture each run's pair set as a value BEFORE the next call overwrites the
        # shared parallel_links.parquet. Both detect_parallels calls return the same
        # path, so reading both paths only after the second run would compare the
        # exhaustive result to itself rather than to the default run (da#172).
        default_pairs = _pair_set(detect_parallels(staging, threshold=0.5))
        exhaustive_pairs = _pair_set(detect_parallels(staging, threshold=0.5, max_block_df=100_000))

        # The default cap loses none of the real intra-cluster pairs relative to the
        # exhaustive scan: C(2,2) * 4 clusters = 4. Anchoring to the independently
        # computed expected set means a default-config regression that drops pairs
        # makes default_pairs diverge from BOTH expected and exhaustive_pairs.
        assert default_pairs == expected
        assert default_pairs == exhaustive_pairs
        assert len(default_pairs) == 4

    def test_corpus_with_no_pairs_warns(self, tmp_path: Path) -> None:
        # Hadiths present but no pair clears the threshold ⇒ "produced nothing" is
        # surfaced as a WARNING so an empty links artifact is not silent (da#160).
        staging = tmp_path / "staging"
        staging.mkdir()
        write_hadiths(
            staging,
            [
                {"source_id": "sunnah:bukhari:1", "matn_en": "alpha beta gamma", "sect": "sunni"},
                {"source_id": "sunnah:bukhari:2", "matn_en": "delta epsilon zeta", "sect": "sunni"},
            ],
            suffix="a",
        )
        with structlog.testing.capture_logs() as logs:
            out = detect_parallels(staging, threshold=0.5)
        assert pq.read_table(out).num_rows == 0
        detected = next(e for e in logs if e["event"] == "parallels_detected")
        assert detected["log_level"] == "warning"
        assert detected["pairs"] == 0
