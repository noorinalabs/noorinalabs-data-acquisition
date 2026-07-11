"""Unit tests for src.graph.prune — canonical-set SHRINK of Narrator nodes (da#413).

These drive the orphan-computation and the guard against a :class:`MockNeo4jClient`,
so they run with no Docker. The graph-truth acceptance (an edge-bearing orphan is
deleted with its edges, a zero-degree canonical narrator survives) needs a real graph
and lives in ``tests/integration/test_prune_narrators.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.graph.prune import (
    SUMMARY_KEYS,
    EmptyCanonicalSetError,
    ExcessiveDeletionError,
    ForeignCanonicalSetError,
    PruneResult,
    UnusableCanonicalSetError,
    prune_narrators,
    read_canonical_ids,
    summary_line,
)
from src.parse.identity import make_canonical_id, make_discriminated_canonical_id
from tests.test_graph.conftest import MockNeo4jClient, write_narrators_canonical


def _graph_rows(ids: list[str]) -> list[dict[str, str]]:
    """Shape of ``MATCH (n:Narrator) RETURN n.id AS id`` records."""
    return [{"id": nid} for nid in ids]


def _write_calls(client: MockNeo4jClient) -> list[dict[str, object]]:
    return [
        params for _query, params in client.calls if isinstance(params, dict) and "batch" in params
    ]


class StatefulFakeNeo4j:
    """A minimal Narrator-set fake that actually applies the DETACH DELETE.

    The shared :class:`MockNeo4jClient` is stateless — its ``execute_read`` always
    returns the same rows — so it cannot exercise the post-delete readback that
    ``deleted`` and the summary ``orphans=`` (post-count) are measured from. This fake
    holds a live id set: reads return the current set, and a batched delete removes
    those ids. Enough to prove the readback semantics without Docker; the real graph
    confirms them in the integration suite.
    """

    def __init__(self, narrator_ids: list[str]) -> None:
        self.narrators: list[str] = list(narrator_ids)
        self.deleted_batches: list[list[str]] = []

    def execute_read(
        self, query: str, parameters: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> list[dict[str, str]]:
        return [{"id": nid} for nid in self.narrators]

    def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        batch = list((parameters or {}).get("batch", []))
        self.deleted_batches.append(batch)
        drop = set(batch)
        self.narrators = [n for n in self.narrators if n not in drop]
        return []


class TestReadCanonicalIds:
    def test_reads_the_canonical_id_set(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:001"}, {"canonical_id": "nar:002"}]
        )
        assert read_canonical_ids(path) == {"nar:001", "nar:002"}

    def test_missing_parquet_refuses(self, curated_dir: Path) -> None:
        missing = curated_dir / "does_not_exist.parquet"
        with pytest.raises(EmptyCanonicalSetError, match="missing"):
            read_canonical_ids(missing)

    def test_unreadable_parquet_refuses(self, curated_dir: Path) -> None:
        # A non-parquet file with the right extension: pq.read_table raises, and the
        # guard folds that into a refusal rather than letting it crash mid-prune.
        bogus = curated_dir / "narrators_canonical.parquet"
        bogus.write_text("this is not a parquet file")
        with pytest.raises(EmptyCanonicalSetError, match="unreadable"):
            read_canonical_ids(bogus)

    def test_empty_parquet_refuses(self, curated_dir: Path) -> None:
        # Present, readable, zero canonical_id values: an empty keep-set would delete
        # every narrator, so it is exactly the catastrophic case the guard exists for.
        path = write_narrators_canonical(curated_dir, [])
        with pytest.raises(EmptyCanonicalSetError, match="zero canonical_id"):
            read_canonical_ids(path)

    def test_blank_ids_do_not_count_as_a_keep_set(self, curated_dir: Path) -> None:
        # Rows present but every canonical_id blank → no usable ids → refuse. (The
        # schema is non-nullable, so blank is the reachable "no id here" value.)
        path = write_narrators_canonical(curated_dir, [{"canonical_id": ""}, {"canonical_id": ""}])
        with pytest.raises(EmptyCanonicalSetError):
            read_canonical_ids(path)


class TestGuardTouchesNothing:
    """The load-bearing safety property: a bad read deletes nothing, reads no graph."""

    @pytest.mark.parametrize("kind", ["missing", "empty", "unreadable"])
    def test_guard_refuses_with_zero_client_calls(self, curated_dir: Path, kind: str) -> None:
        if kind == "missing":
            path = curated_dir / "nope.parquet"
        elif kind == "empty":
            path = write_narrators_canonical(curated_dir, [])
        else:
            path = curated_dir / "narrators_canonical.parquet"
            path.write_text("garbage")

        client = MockNeo4jClient()
        with pytest.raises(EmptyCanonicalSetError):
            prune_narrators(client, path)

        # Neither the read of graph ids nor any DETACH DELETE was issued. The guard
        # fired before the graph was touched at all.
        assert client.calls == [], f"guard ({kind}) reached the graph: {client.calls}"


class TestPruneComputesTheComplement:
    def test_delete_batch_targets_orphan_not_zero_degree_canonical(self, curated_dir: Path) -> None:
        """The whole reason pure-cypher failed, at the set level.

        ``nar:keep_zero_degree`` is a canonical narrator with no edges — the 57,993
        class that a degree prune wrongly deletes. It is in the keep-set, so it is
        NOT in the delete batch. ``nar:orphan`` is absent from the keep-set, so it IS
        — regardless of how many edges it has (DETACH removes them). This asserts the
        batch TARGETING; the real deletion + survival is proven against the stateful
        fake below and on a live graph in the integration suite.
        """
        path = write_narrators_canonical(
            curated_dir,
            [{"canonical_id": "nar:keep_zero_degree"}, {"canonical_id": "nar:keep_busy"}],
        )
        client = MockNeo4jClient()
        client.set_read_results(
            _graph_rows(["nar:keep_zero_degree", "nar:keep_busy", "nar:orphan"])
        )

        result = prune_narrators(client, path)

        writes = _write_calls(client)
        assert writes, "expected a batched DETACH DELETE call"
        deleted_batch = writes[0]["batch"]
        assert deleted_batch == ["nar:orphan"]
        assert "nar:keep_zero_degree" not in deleted_batch
        assert result.orphans_identified == 1

    def test_no_orphans_issues_no_write(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:001"}, {"canonical_id": "nar:002"}]
        )
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:001", "nar:002"]))

        result = prune_narrators(client, path)

        assert result.orphans_identified == 0
        assert _write_calls(client) == []

    def test_batches_respect_batch_size(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        orphans = [f"nar:orphan{i}" for i in range(5)]
        client.set_read_results(_graph_rows(["nar:keep", *orphans]))

        prune_narrators(client, path, batch_size=2)

        batches = [w["batch"] for w in _write_calls(client)]
        assert batches == [orphans[0:2], orphans[2:4], orphans[4:5]]


class TestReadbackSemantics:
    """``deleted`` and the summary ``orphans=`` are measured by reading the graph BACK.

    Driven against the stateful fake so the post-delete state is real, not the pre-read
    echoed. This is where the by-mode ``orphans=`` contract and the pre/post ``deleted``
    are pinned without Docker.
    """

    def test_real_run_deletes_and_post_orphans_is_zero(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:keep_zero"}, {"canonical_id": "nar:keep_busy"}]
        )
        client = StatefulFakeNeo4j(
            ["nar:keep_zero", "nar:keep_busy", "nar:orphan_a", "nar:orphan_b"]
        )

        result = prune_narrators(client, path)

        assert isinstance(result, PruneResult)
        assert result.canonical_ids_seen == 2
        assert result.deleted == 2  # pre(4) - post(2), read back
        assert result.graph_total == 2  # survivors, post
        assert result.orphans == 0  # post-count MUST be 0
        assert result.orphans_identified == 2  # would-delete, pre
        assert result.dry_run is False
        # The graph actually shrank to exactly the canonical survivors.
        assert set(client.narrators) == {"nar:keep_zero", "nar:keep_busy"}

    def test_dry_run_fields_are_pre_state(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan_a", "nar:orphan_b"])

        result = prune_narrators(client, path, dry_run=True)

        assert result.dry_run is True
        assert result.graph_total == 3  # current total, nothing removed
        assert result.orphans == 2  # would-delete count
        assert result.deleted == 0
        assert client.deleted_batches == [], "dry-run issued a delete"
        assert set(client.narrators) == {"nar:keep", "nar:orphan_a", "nar:orphan_b"}


class TestDryRun:
    def test_dry_run_issues_no_write(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:keep", "nar:orphan_a", "nar:orphan_b"]))

        result = prune_narrators(client, path, dry_run=True)

        assert result.dry_run is True
        assert result.orphans_identified == 2  # counted
        assert result.orphans == 2  # summary field: would-delete on a dry run
        assert result.graph_total == 3  # current total (nothing removed)
        assert result.deleted == 0  # but nothing deleted
        assert result.sample == ["nar:orphan_a", "nar:orphan_b"]
        assert _write_calls(client) == [], "dry-run must not issue a DETACH DELETE"

    def test_dry_run_still_reads_the_graph(self, curated_dir: Path) -> None:
        # It reports a real count, so it does read the graph — it just writes nothing.
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = MockNeo4jClient()
        client.set_read_results(_graph_rows(["nar:keep", "nar:orphan"]))

        prune_narrators(client, path, dry_run=True)

        read_calls = [q for q, _p in client.calls if "MATCH (n:Narrator)" in q]
        assert read_calls, "dry-run should have read the graph's narrator ids"


class TestSummaryLine:
    """The one machine-readable line deploy#557's verify greps. Format is a contract."""

    def test_dry_run_line_reports_would_delete(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan_a", "nar:orphan_b"])
        result = prune_narrators(client, path, dry_run=True)
        assert (
            summary_line(result)
            == "PRUNE_NARRATORS_SUMMARY canonical=1 graph_total=3 orphans=2 deleted=0 missing=0"
        )

    def test_real_run_line_reports_zero_orphans_remaining(self, curated_dir: Path) -> None:
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": "nar:keep_a"}, {"canonical_id": "nar:keep_b"}]
        )
        client = StatefulFakeNeo4j(["nar:keep_a", "nar:keep_b", "nar:orphan_a"])
        result = prune_narrators(client, path)
        # canonical=2, graph_total (post survivors)=2, orphans (post)=0, deleted=1, missing=0.
        assert (
            summary_line(result)
            == "PRUNE_NARRATORS_SUMMARY canonical=2 graph_total=2 orphans=0 deleted=1 missing=0"
        )

    def test_missing_is_the_fifth_key_and_comes_last(self, curated_dir: Path) -> None:
        # deploy#557's four-key extraction must be undisturbed by the arrival of a fifth,
        # which is only true if it is APPENDED. Pin the position, not just the presence.
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep"])
        line = summary_line(prune_narrators(client, path, dry_run=True))

        keys = [field.split("=", 1)[0] for field in line.split()[1:]]
        assert keys == list(SUMMARY_KEYS)
        assert keys[-1] == "missing"


class TestSummaryKeyNamespace:
    """The keys are an APPEND-ONLY namespace, because the consumer's regex is unanchored.

    deploy#557's ``summary_field`` extracts a key with a greedy, unanchored ``.*<key>=``
    (da#419). A key whose name *ends with* another key's name is therefore read as that
    other key: ``edges_deleted=`` is silently harvested as ``deleted=``, feeding an edge
    count into a node-count gate. Adding a fifth key is exactly when that bites, so the
    invariant is pinned mechanically here rather than left to whoever adds the sixth.
    """

    def test_no_key_is_a_suffix_of_another(self) -> None:
        for key in SUMMARY_KEYS:
            colliders = [other for other in SUMMARY_KEYS if other != key and other.endswith(key)]
            assert not colliders, (
                f"summary key {key!r} is a suffix of {colliders!r}: the consumer's "
                f"unanchored `.*{key}=` would harvest the wrong field (da#419)"
            )

    def test_summary_line_emits_exactly_the_declared_keys(self, curated_dir: Path) -> None:
        # SUMMARY_KEYS is the thing the suffix invariant above is checked against, so it
        # has to actually BE the line's key set — otherwise the invariant guards a
        # constant nobody emits.
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan"])
        line = summary_line(prune_narrators(client, path))

        assert line.split()[0] == "PRUNE_NARRATORS_SUMMARY"
        emitted = tuple(field.split("=", 1)[0] for field in line.split()[1:])
        assert emitted == SUMMARY_KEYS


# The consumer's extractor, copied VERBATIM from deploy#557 PR#574's
# `.github/workflows/graph-prune-narrators.yml:359-361` at head eef7ea7:
#
#     summary_field() {
#       printf '%s\n' "$1" | sed -n "s/.*PRUNE_NARRATORS_SUMMARY .*$2=\([0-9][0-9]*\).*/\1/p" ...
#
# Re-implementing it in Python `re` would be testing a translation, not the consumer, and
# POSIX BRE is not Python's dialect — so this shells out to the real `sed` with the real
# expression. The producer is the right place for this: it is the producer that breaks the
# consumer by adding a key.
_CONSUMER_SED = r"s/.*PRUNE_NARRATORS_SUMMARY .*{key}=\([0-9][0-9]*\).*/\1/p"


def _consumer_extract(line: str, key: str) -> str:
    """Run deploy#557's actual `summary_field` sed over `line` and return what it reads."""
    proc = subprocess.run(  # noqa: S603
        ["sed", "-n", _CONSUMER_SED.format(key=key)],  # noqa: S607
        input=line + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()[0] if proc.stdout.strip() else ""


@pytest.mark.skipif(shutil.which("sed") is None, reason="needs POSIX sed (the consumer's tool)")
class TestConsumerExtractionCoVerifies:
    """Co-verify the five-key line against deploy#557's REAL sed, not against a paraphrase.

    ``feedback_silent_zero_is_not_a_measurement``: verify the instrument before you read
    it. A test that only asserts "all five keys extract correctly" would pass just as
    happily against a sed that cannot detect a collision at all — so
    :meth:`test_the_harness_can_actually_detect_a_collision` feeds it a line that MUST be
    misread, and requires it to be misread. Only then does the clean read below mean
    anything.
    """

    def test_the_harness_can_actually_detect_a_collision(self) -> None:
        # The positive control. `edges_deleted=` ends with `deleted`, so the unanchored
        # `.*deleted=` harvests 999 — the exact da#419 hazard. If this ever passes as
        # `4`, the harness has stopped reproducing the consumer and every assertion in
        # the sibling test below is vacuous.
        hijacked = (
            "PRUNE_NARRATORS_SUMMARY canonical=1 graph_total=2 orphans=3 "
            "deleted=4 edges_deleted=999"
        )
        assert _consumer_extract(hijacked, "deleted") == "999", (
            "the consumer's regex is supposed to be hijackable by a suffix-colliding key; "
            "if it is not, this harness is no longer the consumer and proves nothing"
        )

    def test_missing_does_not_collide_with_any_existing_key(self, curated_dir: Path) -> None:
        # The reading, now that the instrument is known live. Every key — including the
        # four deploy#557 already extracts — must read its OWN value off the five-key line.
        path = write_narrators_canonical(
            curated_dir, [{"canonical_id": f"nar:{c}"} for c in "abcd"]
        )
        # Five DISTINCT values (4/5/2/0/1), so a key that harvests a neighbour's field
        # shows up as a wrong number rather than coincidentally matching the right one.
        client = StatefulFakeNeo4j(["nar:a", "nar:b", "nar:c", "nar:orphan_x", "nar:orphan_y"])
        line = summary_line(prune_narrators(client, path, dry_run=True))
        assert line == (
            "PRUNE_NARRATORS_SUMMARY canonical=4 graph_total=5 orphans=2 deleted=0 missing=1"
        )

        assert _consumer_extract(line, "canonical") == "4"
        assert _consumer_extract(line, "graph_total") == "5"
        assert _consumer_extract(line, "orphans") == "2"
        assert _consumer_extract(line, "deleted") == "0"
        assert _consumer_extract(line, "missing") == "1"


# --------------------------------------------------------------------------------------
# The stale keep-set: a WRONG-BUT-VALID parquet (da#413 review blocker)
# --------------------------------------------------------------------------------------
#
# The fixture has to be able to PRODUCE the bad state, or the guard it tests is inert and
# we will believe it anyway (`feedback_fixture_makes_guard_assertion_inert`). So the stale
# keep-set below is not a toy and not an empty file: it is built by the SAME production
# minting functions a real resolve run uses, re-keyed by the SAME mechanism that re-keys
# ids between runs.
#
# The mechanism is real. `make_discriminated_canonical_id` (da#337 same-name split) mints
# a DIFFERENT `nar:` id for the same person once the splitter assigns a discriminator, and
# is byte-identical to `make_canonical_id` when it does not (its documented backward-compat
# contract). So a run whose splitter discriminated a narrator, followed by a re-resolve
# whose splitter did not (or chose differently), yields exactly this: a previous canonical
# parquet whose ids are mostly strangers to the live graph — readable, non-empty, schema-
# valid, full of real narrators, and catastrophic if handed to a prune.
#
# These are real narrators from the corpus, not `nar:foo`.
_NARRATOR_NAMES = [
    "ابو هريره",
    "عايشه",
    "عبد الله بن عمر",
    "انس بن مالك",
    "جابر بن عبد الله",
    "ابو سعيد الخدري",
    "عبد الله بن عباس",
    "عمر بن الخطاب",
    "علي بن ابي طالب",
    "محمد بن شهاب الزهري",
]
# The previous run's splitter discriminated the first eight (death-year tags, the shape
# da#337 uses) and left the last two alone.
_PRIOR_RUN_DISCRIMINATORS = {
    "ابو هريره": "d.59",
    "عايشه": "d.58",
    "عبد الله بن عمر": "d.73",
    "انس بن مالك": "d.93",
    "جابر بن عبد الله": "d.78",
    "ابو سعيد الخدري": "d.74",
    "عبد الله بن عباس": "d.68",
    "عمر بن الخطاب": "d.23",
}


def _live_generation_ids() -> list[str]:
    """The ids the LIVE graph was loaded from — this run's minting, undiscriminated."""
    return [make_canonical_id(name) for name in _NARRATOR_NAMES]


def _stale_generation_ids() -> list[str]:
    """The PREVIOUS run's canonical ids: eight re-keyed by the splitter, two unchanged."""
    return [
        make_discriminated_canonical_id(name, _PRIOR_RUN_DISCRIMINATORS.get(name, ""))
        for name in _NARRATOR_NAMES
    ]


def _canonical_parquet(curated: Path, ids: list[str], names: list[str] | None = None) -> Path:
    """A realistic narrators_canonical.parquet — full rows, not bare ids."""
    names = names or _NARRATOR_NAMES
    return write_narrators_canonical(
        curated,
        [
            {
                "canonical_id": cid,
                "name_ar": name,
                "name_ar_normalized": name,
                "mention_count": 100 + i,
                "source_corpora": ["sunnah"],
            }
            for i, (cid, name) in enumerate(zip(ids, names, strict=True))
        ],
    )


_ORPHANS = ["nar:orphan_a", "nar:orphan_b"]


class TestStaleKeepSetIsRefused:
    """The blocker: nothing bound the keep-set to the graph it prunes (Jean-Claude, #414).

    A stale ``parquet_ref`` is the single most plausible operator error, and it is not
    self-announcing. It passes every guard this command had, and — the reason it needed
    a guard in *this* layer rather than a gate in the consumer's — it passes every
    post-op check too, because they are all derived from the same keep-set they would be
    validating. They prove the prune obeyed the parquet. None can ask if it was the right
    parquet.
    """

    def test_the_stale_keep_set_passes_every_pre_existing_guard(self, curated_dir: Path) -> None:
        """First: prove the fixture really does reach the bad state.

        If the empty/missing/unreadable guard already caught this, the new guard would be
        guarding nothing and the tests below would be theatre. It does not: the stale
        parquet is readable, non-empty, and yields a full ten-id keep-set.
        """
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())

        keep_set = read_canonical_ids(stale)  # does NOT raise

        assert len(keep_set) == 10
        assert all(cid.startswith("nar:") for cid in keep_set)
        # And it is genuinely a DIFFERENT set from the graph's: eight of its ten ids were
        # re-keyed by the splitter, so the live graph has never seen them.
        live = set(_live_generation_ids())
        assert len(keep_set - live) == 8
        assert len(keep_set & live) == 2  # the two the previous run left undiscriminated

    def test_without_the_ceiling_the_stale_keep_set_deletes_the_live_generation(
        self, curated_dir: Path
    ) -> None:
        """The red. This is what the code did before the guard, and what it still does if
        an operator lifts the ceiling — so the damage is measured, not asserted.

        Eight legitimate, edge-bearing, current-generation narrators are DETACH DELETEd
        because a stale parquet does not list them. Every gate downstream would go green.
        """
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        live = _live_generation_ids()
        client = StatefulFakeNeo4j([*live, *_ORPHANS])

        # max_missing_fraction=1.0 disables the ceiling — i.e. the pre-fix behaviour.
        result = prune_narrators(client, stale, max_missing_fraction=1.0)

        # 8 legitimate narrators destroyed, plus the 2 real orphans: 10 of 12 gone.
        assert result.deleted == 10
        assert len(client.narrators) == 2
        for name in _NARRATOR_NAMES[:8]:
            assert make_canonical_id(name) not in client.narrators, (
                f"{name} is a live, canonical narrator and was deleted by a stale keep-set"
            )
        # And here is the whole point: EVERY post-op gate still passes. The graph was not
        # emptied, no orphan remains, and deleted equals what the dry run predicted. A
        # workflow reading only these would print "post-op verification PASSED".
        assert result.graph_total > 0  # gate A: not emptied
        assert result.orphans == 0  # gate D: no orphan remains
        assert result.deleted == result.orphans_identified  # gate C: matches prediction
        # Only `missing` separates this from a correct run. It is 8.
        assert result.missing == 8

    def test_stale_keep_set_is_refused_before_any_write(self, curated_dir: Path) -> None:
        """The green. Default ceiling: refused, and NOT ONE ROW DELETED."""
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        live = _live_generation_ids()
        client = StatefulFakeNeo4j([*live, *_ORPHANS])
        before = list(client.narrators)

        with pytest.raises(ForeignCanonicalSetError) as raised:
            prune_narrators(client, stale)

        # Zero deletion. The graph is exactly as it was — the load-bearing guarantee.
        assert client.deleted_batches == [], "a refused prune issued a DETACH DELETE"
        assert client.narrators == before
        # The refusal reports the signal that produced it, so the operator can act.
        assert raised.value.missing == 8
        assert raised.value.canonical_total == 10
        assert "does not belong to this graph" in str(raised.value)

    def test_refusal_fires_on_a_dry_run_too(self, curated_dir: Path) -> None:
        # The operator's mandatory dry run is precisely where a wrong parquet should
        # surface — refusing only on the real run would let the dry run bless it.
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        client = StatefulFakeNeo4j([*_live_generation_ids(), *_ORPHANS])

        with pytest.raises(ForeignCanonicalSetError):
            prune_narrators(client, stale, dry_run=True)

        assert client.deleted_batches == []

    def test_the_guard_is_catchable_as_the_cli_catches_it(self, curated_dir: Path) -> None:
        # _cmd_prune_narrators catches UnusableCanonicalSetError and exits
        # UNUSABLE_CANONICAL_SET. If this door were not under that supertype it would escape
        # to CPython's handler and exit 1 (LOAD_FAILED) — "the load failed", for a command
        # that deleted nothing and is not a load.
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        client = StatefulFakeNeo4j(_live_generation_ids())

        with pytest.raises(UnusableCanonicalSetError):
            prune_narrators(client, stale)


class TestMissingIsReportedNotAsserted:
    """The dual. A guard that refuses every non-zero ``missing`` trades one failure for
    another: :attr:`REFUSED_ROWS`, :attr:`STOPPED_AT_LIMIT` and a nodes-only load each
    leave canonical ids legitimately unloaded, and a hard ``missing == 0`` in the CLI
    would refuse a valid prune after any of them. The CLI reports; deploy#557 owns the
    refuse-or-proceed policy.
    """

    def test_correct_keep_set_complete_load_reports_zero(self, curated_dir: Path) -> None:
        # The da#352 property: for the parquet the graph was loaded from, every canonical
        # id is present in the graph (129,234 = 71,241 edged + 57,993 zero-degree).
        correct = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j([*_live_generation_ids(), *_ORPHANS])

        result = prune_narrators(client, correct)

        assert result.missing == 0
        assert result.deleted == 2  # exactly the orphans
        assert set(client.narrators) == set(_live_generation_ids())

    def test_correct_keep_set_partial_load_is_reported_and_proceeds(
        self, curated_dir: Path
    ) -> None:
        """A legitimately partial load must still be PRUNABLE, and its ``missing`` visible.

        The graph was loaded from this very parquet, but two canonical narrators never
        made it in (REFUSED_ROWS). ``missing`` is 2 — non-zero, correctly so — and the
        prune proceeds: it is well under any plausible magnitude, and the CLI is not the
        layer that decides whether a partial load may be pruned.
        """
        correct = _canonical_parquet(curated_dir, _live_generation_ids())
        live = _live_generation_ids()
        loaded = live[:8]  # two canonical narrators refused at load time
        client = StatefulFakeNeo4j([*loaded, *_ORPHANS])

        result = prune_narrators(client, correct)  # default ceiling: NO refusal

        assert result.missing == 2  # reported…
        assert result.canonical_ids_seen == 10
        assert result.deleted == 2  # …and exactly the orphans went
        assert result.orphans == 0
        assert set(client.narrators) == set(loaded), "a partial load's narrators were pruned"
        assert "missing=2" in summary_line(result)

    def test_the_two_classes_are_separated_by_missing(
        self, curated_dir: Path, tmp_path: Path
    ) -> None:
        """The separation, side by side — a number for one class alone proves nothing.

        Same graph, same command, two keep-sets. The four keys deploy#557 already reads
        cannot tell them apart on the thing that matters; ``missing`` can.
        """
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        correct = _canonical_parquet(good_dir, _live_generation_ids())
        stale = _canonical_parquet(bad_dir, _stale_generation_ids())
        graph = [*_live_generation_ids(), *_ORPHANS]

        good = prune_narrators(StatefulFakeNeo4j(list(graph)), correct, dry_run=True)
        bad = prune_narrators(
            StatefulFakeNeo4j(list(graph)), stale, dry_run=True, max_missing_fraction=1.0
        )

        # Both keep-sets are the same SIZE and both "work". The correct one would delete
        # the 2 orphans; the stale one would delete 10 nodes — but a workflow cannot know
        # 10 is wrong without knowing the right answer, which is the whole problem.
        assert good.canonical_ids_seen == bad.canonical_ids_seen == 10
        assert good.graph_total == bad.graph_total == 12

        # `missing` separates them cleanly, and it is the ONLY field that does.
        assert good.missing == 0
        assert bad.missing == 8


# --------------------------------------------------------------------------------------
# The TRUNCATED keep-set: same generation, right ids, just too few (da#413 2nd review)
# --------------------------------------------------------------------------------------
#
# `missing` bounds `canonical - graph` — a set whose elements are NOT in the graph and so
# cannot be deleted. It is structurally incapable of bounding the set that DOES delete,
# `orphans = graph - canonical`. A truncated keep-set walks straight through that gap: a
# resolve stopped early, a half-written parquet, a partial upload, a null-heavy
# canonical_id column. Every id in it is real and current, so `missing` is EXACTLY 0 and
# every id-provenance instrument — here and in deploy#574 — reports the keep-set healthy,
# while the prune deletes all but the remnant.
#
# These fixtures work at generation scale rather than with a handful of ids, because the
# guard is a FRACTION and a 3-node graph cannot express a realistic one. Names are composed
# from real base names and real nisbas, and every id is minted by the production
# `make_canonical_id` — the same discipline as the stale-keep-set fixtures above.
_NISBAS = [
    "الزهري",
    "البصري",
    "الكوفي",
    "المدني",
    "المكي",
    "الشامي",
    "اليماني",
    "المصري",
    "البغدادي",
    "النيسابوري",
    "الطبري",
    "الرازي",
    "الاصبهاني",
    "الدمشقي",
    "الحمصي",
    "الواسطي",
    "الانصاري",
    "التميمي",
    "القرشي",
    "الهمداني",
    "الجعفي",
    "السلمي",
    "الاسدي",
    "العبدي",
    "الثقفي",
    "الخزاعي",
    "النخعي",
    "المزني",
    "الغفاري",
    "الجهني",
    "الطائي",
    "العبسي",
    "الفزاري",
    "الضبي",
    "السدوسي",
    "الحنفي",
    "الازدي",
    "الخولاني",
    "الحضرمي",
    "المرادي",
]


def _generation_names(count: int) -> list[str]:
    """`count` real-shaped narrator names — base name + nisba, the actual naming morphology."""
    names = [f"{base} {nisba}" for nisba in _NISBAS for base in _NARRATOR_NAMES]
    assert count <= len(names), f"only {len(names)} distinct names available"
    return names[:count]


def _generation_ids(count: int, offset: int = 0) -> list[str]:
    """`count` canonical ids minted by the PRODUCTION function from real-shaped names."""
    return [make_canonical_id(n) for n in _generation_names(count + offset)[offset:]]


class TestTruncatedKeepSetIsRefused:
    """The second door to the same catastrophe (Kavitha Sundaramurthy + Jean-Claude, #414).

    The keep-set is not foreign. It is this graph's own — every id real, every id current.
    It is merely INCOMPLETE, and incompleteness is invisible to every instrument that asks
    whether the keep-set's ids belong to this graph. `missing` is the wrong axis for it,
    by construction, so the delete magnitude has to be bounded on its own.
    """

    def test_the_truncated_keep_set_passes_every_pre_existing_guard(
        self, curated_dir: Path
    ) -> None:
        """The control, and the whole reason the new guard is not theatre.

        Kavitha asked for exactly this: prove the fixture sails through everything that
        already exists — INCLUDING `missing`, which is not merely small here but exactly
        **zero**. Without this control, the new ceiling could be guarding a state nothing
        can reach, and we would believe it. (`feedback_fixture_makes_guard_assertion_inert`.)
        """
        live = _generation_ids(200)
        graph = [*live, *_ORPHANS]  # 202 nodes: a real generation + 2 real orphans
        truncated = _canonical_parquet(  # a resolve that stopped after 10 rows
            curated_dir, live[:10], names=_generation_names(10)
        )

        # (1) the empty / missing / unreadable guard: passes — readable, 10 real ids.
        keep_set = read_canonical_ids(truncated)
        assert len(keep_set) == 10

        # (2) `missing` is EXACTLY 0 — every id in the keep-set really is in the graph.
        #     So the ForeignCanonicalSetError ceiling passes, AND deploy#574's planned
        #     `missing == 0` policy does not merely miss this: it BLESSES it.
        graph_set = set(graph)
        missing = sum(1 for cid in keep_set if cid not in graph_set)
        assert missing == 0, "the truncated keep-set must be same-generation, or it proves nothing"

        # (3) the consumer's whole-graph-wipe check (`orphans >= graph_pre`) passes too:
        #     10 nodes survive, so it is not a total wipe by that test's reckoning.
        orphans = [g for g in graph if g not in keep_set]
        assert len(orphans) < len(graph)

        # (4) and Weronika's proposed invariant, EXPECTED_POST == CANON_N, also passes.
        assert len(graph) - len(orphans) == len(keep_set)

        # Every guard we have ever designed says this keep-set is fine. It would delete:
        assert len(orphans) / len(graph) > 0.95  # 95%+ of the narrator graph

    def test_without_the_ceiling_a_truncated_keep_set_wipes_the_graph(
        self, curated_dir: Path
    ) -> None:
        """The red: measured damage, not asserted danger."""
        live = _generation_ids(200)
        client = StatefulFakeNeo4j([*live, *_ORPHANS])
        truncated = _canonical_parquet(curated_dir, live[:10], names=_generation_names(10))

        result = prune_narrators(client, truncated, max_orphan_fraction=1.0)

        assert result.missing == 0  # every provenance instrument stays silent
        assert result.deleted == 192  # 202 - 10
        assert len(client.narrators) == 10
        assert result.deleted / 202 > 0.95
        # And every post-op gate greens, exactly as with the stale keep-set.
        assert result.orphans == 0
        assert result.deleted == result.orphans_identified
        assert result.graph_total > 0

    def test_truncated_keep_set_is_refused_before_any_write(self, curated_dir: Path) -> None:
        """The green: refused by the delete-side ceiling, ZERO rows deleted."""
        live = _generation_ids(200)
        client = StatefulFakeNeo4j([*live, *_ORPHANS])
        before = list(client.narrators)
        truncated = _canonical_parquet(curated_dir, live[:10], names=_generation_names(10))

        with pytest.raises(ExcessiveDeletionError) as raised:
            prune_narrators(client, truncated)

        assert client.deleted_batches == [], "a refused prune issued a DETACH DELETE"
        assert client.narrators == before
        assert raised.value.orphans == 192
        assert raised.value.graph_total == 202
        # The refusal names the trap explicitly: a clean `missing` does NOT clear this.
        assert raised.value.missing == 0
        assert "TRUNCATED" in str(raised.value)

    def test_refusal_fires_on_a_dry_run_too(self, curated_dir: Path) -> None:
        live = _generation_ids(200)
        client = StatefulFakeNeo4j([*live, *_ORPHANS])
        truncated = _canonical_parquet(curated_dir, live[:10], names=_generation_names(10))

        with pytest.raises(ExcessiveDeletionError):
            prune_narrators(client, truncated, dry_run=True)

        assert client.deleted_batches == []

    def test_the_guard_is_catchable_as_the_cli_catches_it(self, curated_dir: Path) -> None:
        live = _generation_ids(200)
        client = StatefulFakeNeo4j([*live, *_ORPHANS])
        truncated = _canonical_parquet(curated_dir, live[:10], names=_generation_names(10))

        with pytest.raises(UnusableCanonicalSetError):
            prune_narrators(client, truncated)


class TestLargeLegitimatePruneStillProceeds:
    """The dual — and the reason the ceiling is 0.9 rather than the intuitive 0.5.

    This subcommand EXISTS to clean up after a mass id re-mint (da#356/da#376). The graph
    is the union of every load ever run against it, so that cleanup legitimately orphans
    the entire previous generation — well over half the graph. A ceiling near one half
    refuses this command's own primary use case, the operator overrides, the override
    becomes reflex, and the guard is dead. Jean-Claude caught this; it is pinned here so
    nobody "tightens" the ceiling later and quietly breaks the tool.
    """

    def test_full_regeneration_orphans_the_majority_and_still_proceeds(
        self, curated_dir: Path
    ) -> None:
        # The accumulated graph: an old generation (200) still resident, plus a freshly
        # loaded, fully re-minted new one (150). The keep-set is the new generation — it
        # is CORRECT, and it orphans every node of the old one.
        old_generation = _generation_ids(200)
        new_generation = _generation_ids(150, offset=200)
        assert not (set(old_generation) & set(new_generation)), "re-mint must change the ids"

        client = StatefulFakeNeo4j([*old_generation, *new_generation])
        correct = _canonical_parquet(
            curated_dir, new_generation, names=_generation_names(350)[200:]
        )

        result = prune_narrators(client, correct)  # default ceiling: NO refusal

        # 200 of 350 = 57.1% of the graph deleted — a majority — and it is CORRECT.
        assert result.deleted == 200
        assert result.deleted / 350 > 0.5, "the whole point: a legit prune CAN exceed half"
        assert result.missing == 0
        assert result.orphans == 0
        assert set(client.narrators) == set(new_generation)

    def test_a_ceiling_at_one_half_would_have_refused_that(self, curated_dir: Path) -> None:
        # Pin the counterfactual, so the choice of 0.9 is defended by a test and not only
        # by a comment. This is the prune the intuitive ceiling would have destroyed.
        old_generation = _generation_ids(200)
        new_generation = _generation_ids(150, offset=200)
        client = StatefulFakeNeo4j([*old_generation, *new_generation])
        correct = _canonical_parquet(
            curated_dir, new_generation, names=_generation_names(350)[200:]
        )

        with pytest.raises(ExcessiveDeletionError):
            prune_narrators(client, correct, max_orphan_fraction=0.5)

        assert client.deleted_batches == []


class TestOrphanCeiling:
    def test_at_the_ceiling_is_allowed_above_it_refuses(self, curated_dir: Path) -> None:
        # Strictly-greater, same as the missing ceiling. graph=10, orphans=9 -> exactly 0.9.
        live = _generation_ids(10)
        keep = live[:1]
        client = StatefulFakeNeo4j(list(live))
        path = _canonical_parquet(curated_dir, keep, names=_generation_names(1))

        at_ceiling = prune_narrators(client, path, dry_run=True, max_orphan_fraction=0.9)
        assert at_ceiling.orphans == 9  # dry-run: the would-delete count

        with pytest.raises(ExcessiveDeletionError):
            prune_narrators(client, path, dry_run=True, max_orphan_fraction=0.89)

    def test_empty_graph_does_not_divide_by_zero(self, curated_dir: Path) -> None:
        # Nothing to delete, no fraction to speak of. The `missing` guard owns the empty
        # graph (it refuses it by default); this one must simply not explode when the
        # operator has deliberately waived that one.
        path = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j([])

        result = prune_narrators(client, path, dry_run=True, max_missing_fraction=1.0)

        assert result.graph_total == 0
        assert result.orphans == 0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_fraction_raises(self, curated_dir: Path, bad: float) -> None:
        path = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j(_live_generation_ids())
        with pytest.raises(ValueError, match=r"max_orphan_fraction.*\[0.0, 1.0\]"):
            prune_narrators(client, path, max_orphan_fraction=bad)


class TestMissingCeiling:
    def test_at_the_ceiling_is_allowed_above_it_refuses(self, curated_dir: Path) -> None:
        # Strictly-greater: the ceiling is the last ACCEPTABLE value, not the first
        # rejected one. Pinned because "> vs >=" is exactly the kind of thing a later
        # refactor flips without noticing.
        correct = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j(_live_generation_ids()[:5])  # missing = 5 of 10 = 0.5

        at_ceiling = prune_narrators(client, correct, dry_run=True, max_missing_fraction=0.5)
        assert at_ceiling.missing == 5

        with pytest.raises(ForeignCanonicalSetError):
            prune_narrators(client, correct, dry_run=True, max_missing_fraction=0.49)

    def test_fraction_of_one_disables_the_ceiling(self, curated_dir: Path) -> None:
        # The documented escape hatch: even a keep-set with ZERO overlap is allowed
        # through at 1.0, because the operator asked for it explicitly.
        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        client = StatefulFakeNeo4j([])  # empty graph: missing == canonical

        result = prune_narrators(client, stale, dry_run=True, max_missing_fraction=1.0)

        assert result.missing == 10
        assert result.graph_total == 0

    def test_empty_graph_is_refused_by_default(self, curated_dir: Path) -> None:
        # A graph holding none of the keep-set's ids is the most-wrong case there is, and
        # "the graph is empty" is the correct diagnosis to hand the operator — not a
        # cheerful no-op that reports zero orphans and zero deletions.
        correct = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j([])

        with pytest.raises(ForeignCanonicalSetError, match="100.0%"):
            prune_narrators(client, correct)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_fraction_raises(self, curated_dir: Path, bad: float) -> None:
        correct = _canonical_parquet(curated_dir, _live_generation_ids())
        client = StatefulFakeNeo4j(_live_generation_ids())
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            prune_narrators(client, correct, max_missing_fraction=bad)


class TestNoWriteIntoCanonicalDir:
    """deploy#557 mounts the parquet dir read-only: the subcommand must not write there.

    Unlike ``load`` (which writes a manifest + audit entry into the data dir and trips
    the ``:ro`` mount, da#348), ``prune-narrators`` reads the parquet and writes ONLY to
    the graph. Proven structurally: no new file appears beside the canonical parquet on
    either a dry run or a real run.
    """

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_no_side_file_written_next_to_canonical(self, curated_dir: Path, dry_run: bool) -> None:
        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        before = sorted(p.name for p in curated_dir.iterdir())
        client = StatefulFakeNeo4j(["nar:keep", "nar:orphan"])

        prune_narrators(client, path, dry_run=dry_run)

        after = sorted(p.name for p in curated_dir.iterdir())
        assert after == before == ["narrators_canonical.parquet"]


class TestCliEmitsSummaryLine:
    """deploy#557's verify HARD-FAILS if the summary line is absent from stdout, so the
    CLI must always print it. Stub the DB layer and assert the line reaches stdout."""

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_cmd_prints_summary_line(
        self,
        curated_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        dry_run: bool,
    ) -> None:
        import src.cli as cli

        path = write_narrators_canonical(curated_dir, [{"canonical_id": "nar:keep"}])
        canned = PruneResult(
            canonical_ids_seen=1,
            graph_total=1,
            orphans=0,
            deleted=3,
            dry_run=dry_run,
            orphans_identified=3,
            missing=0,
            sample=["nar:orphan"],
        )

        class _DummyClient:
            def __enter__(self) -> _DummyClient:
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        def _canned_prune(
            _client: object, _p: Path, **_kwargs: object
        ) -> PruneResult:  # accepts whatever ceilings the CLI passes
            return canned

        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", lambda *_a, **_k: _DummyClient())
        monkeypatch.setattr("src.graph.prune.prune_narrators", _canned_prune)

        cli._cmd_prune_narrators(canonical=str(path), dry_run=dry_run)

        out = capsys.readouterr().out
        assert (
            "PRUNE_NARRATORS_SUMMARY canonical=1 graph_total=1 orphans=0 deleted=3 missing=0" in out
        )


class TestCliRefusesStaleKeepSet:
    """End to end through the CLI: a stale --canonical exits 12 with the graph untouched.

    The unit tests above prove `prune_narrators` raises. This proves the CLI *catches* the
    new door — it catches the supertype, so a subclass that escaped would exit 1
    (LOAD_FAILED, "the load failed") for a command that is not a load and deleted nothing.
    """

    def test_stale_canonical_exits_empty_canonical_set(
        self,
        curated_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import src.cli as cli
        from src.exit_codes import ExitCode

        stale = _canonical_parquet(curated_dir, _stale_generation_ids())
        client = StatefulFakeNeo4j([*_live_generation_ids(), *_ORPHANS])

        class _DummyClient:
            def __enter__(self) -> StatefulFakeNeo4j:
                return client

            def __exit__(self, *_a: object) -> None:
                return None

        monkeypatch.setattr(cli, "_check_neo4j", lambda: None)
        monkeypatch.setattr("src.utils.neo4j_client.Neo4jClient", lambda *_a, **_k: _DummyClient())

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_prune_narrators(canonical=str(stale))

        assert exit_info.value.code == ExitCode.UNUSABLE_CANONICAL_SET
        assert client.deleted_batches == [], "the CLI deleted rows before refusing"
        err = capsys.readouterr().err
        assert "does not belong to this graph" in err
