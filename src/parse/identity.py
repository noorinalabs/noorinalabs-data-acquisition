"""Canonical entity-identity contract for the ingest pipeline.

This module is the **single source of truth** for how a ``source_id`` and every
graph node id is constructed, across *all* ingest paths (the batch
``src/graph`` loaders here, and the streaming/normalize path in
``noorinalabs-isnad-ingest-platform``). It is the keystone for da#82 / epic
da#81: every per-source light-up builds to the grammar defined here.

Grammar
-------
``source_id`` (staging key, ``HADITH_SCHEMA.source_id``)::

    <corpus>:<collection>[:<part>...]

* ``<corpus>``     one of :data:`SOURCE_CORPORA` (the ``SourceCorpus`` enum) —
  the namespace that makes ids collision-safe across sources.
* ``<collection>`` per-corpus collection slug (e.g. ``"bukhari"``).
* ``<part>...``    positional disambiguators (``book:chapter:ordinal``, or a
  per-row index for dataset sources). The corpus appears **exactly once**, as
  the leading segment.

Graph node ids = ``<type-prefix><bare-id>``::

    Hadith         hdt:<source_id>
    Collection     col:<corpus>:<collection>
    Narrator       nar:<canonical-id>
    Chain          chn:<source_id>-<index>
    Grading        grd:<source_id>

Two historical identity hazards this module designs out
-------------------------------------------------------
1. **Double-prefix** (main#139): the streaming/normalize path prepended the
   corpus a second time (``hdt:sunnah:sunnah:...``) while the batch loader did
   not, so the *same* hadith became two graph nodes (toy ``h-1`` fixtures masked
   it). :func:`hadith_node_id` is the ONE prefixing rule both paths call: it is
   idempotent on the ``hdt:`` prefix **and** collapses an accidentally doubled
   leading corpus, so both paths converge on exactly one id per hadith.
2. **collection-ref vs in-book-ordinal** (da#77): ``source_id``'s positional
   tail is a stable within-collection key. The *in-book ordinal* that flows to
   ``APPEARS_IN.hadith_number_in_book`` is the staging ``hadith_number`` column —
   **not** the collection-wide reference number (sunnah.com exposes both). See
   ``HADITH_SCHEMA`` and ``src/graph/load_edges.py`` ``_APPEARS_IN_QUERY``.

Free-text provenance keys are NOT corpus ids (da#89)
----------------------------------------------------
:func:`base.generate_source_id` is for hadith ``source_id``\\ s only: its first
argument MUST be a :data:`SOURCE_CORPORA` value and it fails fast otherwise. A
bio/narrator provenance key is namespaced by its *bio source* (a free-text label
like ``"kaggle_narrators"``), which is **not** a hadith ``SourceCorpus`` — so
build it with ``ID_DELIMITER.join([...])`` directly, never with
``generate_source_id`` (see ``sanadset.py`` ``bio_id``). Passing a non-corpus
first arg to ``generate_source_id`` aborted the whole parse at runtime and was
masked by unit tests with no raw dir (the da#89 crash). ``scripts/
check_source_id_corpus.py`` (test: ``tests/test_parse/
test_source_id_corpus_guard.py``) is a CI static guard that asserts every
``generate_source_id`` call-site passes a statically-provable ``SourceCorpus``
first arg, blocking reintroduction.
"""

from __future__ import annotations

import uuid

from src.models.enums import SourceCorpus
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "SOURCE_CORPORA",
    "HADITH_ID_PREFIX",
    "COLLECTION_ID_PREFIX",
    "NARRATOR_ID_PREFIX",
    "CHAIN_ID_PREFIX",
    "GRADING_ID_PREFIX",
    "ID_DELIMITER",
    "apply_prefix",
    "bare_source_id",
    "hadith_node_id",
    "collection_node_id",
    "narrator_node_id",
    "chain_node_id",
    "grading_node_id",
    "is_double_prefixed",
    "validate_source_id",
    "CANONICAL_NAMESPACE",
    "make_canonical_id",
    "make_discriminated_canonical_id",
]

# Unit-separator (U+001F) joining a normalized name to a da#337 discriminator when
# minting a *discriminated* canonical id. It cannot occur in a normalized name (the
# Arabic normalizer strips control/format marks and collapses whitespace), so a
# discriminator can never collide with a name that merely happens to contain the
# joiner — see :func:`make_discriminated_canonical_id`.
_DISCRIMINATOR_SEP = "\x1f"

# Fixed UUID5 namespace for canonical Narrator ids. A canonical narrator's id is
# ``nar:<uuid5(CANONICAL_NAMESPACE, normalized-name)>`` — deterministic from the
# narrator's normalized name so every producer (the mention-driven disambiguator
# in ``src/resolve/disambiguate.py`` and the bio-direct promoter in
# ``src/resolve/bio_promote.py``) that sees the same name converges on the same
# Narrator node. This namespace value is load-bearing: changing it re-keys every
# existing canonical narrator, so it is pinned here as the single source of truth.
CANONICAL_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# The corpus namespace — derived from the enum so the two never drift. Note that
# ``sunnah_scraped`` intentionally shares the ``sunnah`` corpus (the scraper and
# the API describe the same collections, so a shared namespace lets the same
# hadith dedup to one node).
SOURCE_CORPORA: frozenset[str] = frozenset(c.value for c in SourceCorpus)

ID_DELIMITER = ":"

# Running count of double-corpus-prefix collapses this process, used to SAMPLE the
# per-collapse warning in ``_collapse_double_corpus`` (it fires on a path called
# millions of times; #723). Process-local; not reset between calls by design.
_double_corpus_collapse_count = 0

HADITH_ID_PREFIX = "hdt:"
COLLECTION_ID_PREFIX = "col:"
NARRATOR_ID_PREFIX = "nar:"
CHAIN_ID_PREFIX = "chn:"
GRADING_ID_PREFIX = "grd:"

# Every graph type-prefix, longest-first so a stripping pass is unambiguous.
_TYPE_PREFIXES = (
    HADITH_ID_PREFIX,
    COLLECTION_ID_PREFIX,
    NARRATOR_ID_PREFIX,
    CHAIN_ID_PREFIX,
    GRADING_ID_PREFIX,
)


def apply_prefix(value: str, prefix: str) -> str:
    """Idempotently prepend *prefix* to *value*.

    Replaces the ``f"{p}{v}" if not v.startswith(p) else v`` idiom that was
    duplicated across every loader. Idempotent: applying the same prefix twice
    yields the same string.
    """
    return value if value.startswith(prefix) else f"{prefix}{value}"


def _collapse_double_corpus(body: str) -> str:
    """Collapse a doubled (or N-times repeated) leading corpus segment.

    ``"sunnah:sunnah:bukhari:1"`` -> ``"sunnah:bukhari:1"``. Only collapses when
    the repeated leading segment is a *known* corpus, so legitimate repeated
    slugs deeper in an id (or a non-corpus first segment) are never touched.
    Emits a warning when it collapses — a collapse means an upstream producer
    bypassed the canonical builder (the main#139 streaming-path bug), which the
    operator should still see.
    """
    segments = body.split(ID_DELIMITER)
    collapsed = 0
    while len(segments) >= 2 and segments[0] in SOURCE_CORPORA and segments[0] == segments[1]:
        segments.pop(0)
        collapsed += 1
    if collapsed:
        global _double_corpus_collapse_count
        _double_corpus_collapse_count += 1
        # This fires on the hot id-canonicalization path — every hadith/chain/
        # grading id build, and the PARALLEL_OF loader calls it 13M+ times. The
        # sanadset double-prefix bug (main#139) trips it on millions of rows: the
        # #723 stage reload emitted 8.45M of these warnings (a 4.3 GB container
        # log) and the per-call structlog rendering measurably slowed the load.
        # Sample it — surface the condition once, then every 100k-th, with the
        # running magnitude in ``occurrences`` — instead of one line per record.
        if _double_corpus_collapse_count == 1 or _double_corpus_collapse_count % 100_000 == 0:
            logger.warning(
                "collapsed_double_corpus_prefix",
                original=body,
                canonical=ID_DELIMITER.join(segments),
                collapsed_segments=collapsed,
                occurrences=_double_corpus_collapse_count,
            )
    return ID_DELIMITER.join(segments)


def bare_source_id(value: str) -> str:
    """Return the canonical *bare* ``source_id`` for *value*.

    Strips a leading ``hdt:`` graph prefix if present, then collapses any
    doubled leading corpus. This is the single canonicalization point shared by
    :func:`hadith_node_id`, :func:`chain_node_id` and :func:`grading_node_id`,
    so every hadith-derived id agrees on one identity for a given hadith.
    """
    body = value[len(HADITH_ID_PREFIX) :] if value.startswith(HADITH_ID_PREFIX) else value
    return _collapse_double_corpus(body)


def hadith_node_id(source_id: str) -> str:
    """Canonical Hadith graph node id for a staging ``source_id``.

    THE one prefixing rule for Hadith nodes — every ingest path (batch loaders,
    streaming normalize) must route through this so the same hadith yields
    exactly one node id. Idempotent, and collapses the main#139 double-prefix.
    """
    return f"{HADITH_ID_PREFIX}{bare_source_id(source_id)}"


def collection_node_id(collection_id: str) -> str:
    """Canonical Collection graph node id (``col:<corpus>:<collection>``)."""
    return apply_prefix(collection_id, COLLECTION_ID_PREFIX)


def narrator_node_id(canonical_id: str) -> str:
    """Canonical Narrator graph node id (``nar:<canonical-id>``)."""
    return apply_prefix(canonical_id, NARRATOR_ID_PREFIX)


def make_canonical_id(name_normalized: str) -> str:
    """Deterministic canonical Narrator id from a *normalized* name.

    Returns ``nar:<uuid5(CANONICAL_NAMESPACE, name_normalized)>``. THE one rule
    for minting a canonical narrator id — both the mention-driven disambiguator
    (``src/resolve/disambiguate.py``) and the bio-direct promoter
    (``src/resolve/bio_promote.py``) route through it so the same normalized name
    always maps to the same Narrator node (and the ``nar:`` prefix the graph
    loader ``load_nodes._load_narrators`` validates on import). The input must be
    pre-normalized (see ``src.utils.arabic.normalize_arabic``) so equivalent
    spellings collapse to one id.
    """
    return narrator_node_id(str(uuid.uuid5(CANONICAL_NAMESPACE, name_normalized)))


def make_discriminated_canonical_id(name_normalized: str, discriminator: str = "") -> str:
    """Canonical Narrator id from a normalized name, optionally *discriminated*.

    The sibling rule to :func:`make_canonical_id` for the da#337 same-name split
    stage. Because :func:`make_canonical_id` keys purely on the normalized name,
    distinct people who share a normalized name (bare kunyas like ``أبو عبد الله``,
    single-token mononyms, captured name fragments) collapse into ONE ``nar:`` node
    and inflate its betweenness centrality. This helper lets that stage mint a
    *distinct* id per referent by folding a ``discriminator`` (e.g. a
    death-year/generation/nisba tag) into the uuid5 input.

    Backward-compat contract: an **empty** (falsy) ``discriminator`` returns a
    result **byte-identical** to ``make_canonical_id(name_normalized)`` — the
    undiscriminated id every existing producer already mints — so wiring this in
    re-keys nothing for the un-split majority. Only a non-empty discriminator
    diverges, minting ``nar:<uuid5(CANONICAL_NAMESPACE, name + SEP + discriminator)>``.

    The name and discriminator are joined by the :data:`_DISCRIMINATOR_SEP` unit
    separator (U+001F), which a normalized name can never contain, so a
    discriminator can never collide with name content — ``(name="a\\x1fb", disc="")``
    and ``(name="a", disc="b")`` mint different ids even though a bare ``+`` join
    would conflate them. The input name must be pre-normalized
    (see ``src.utils.arabic.normalize_arabic``), exactly as for
    :func:`make_canonical_id`.
    """
    if not discriminator:
        return make_canonical_id(name_normalized)
    key = name_normalized + _DISCRIMINATOR_SEP + discriminator
    return narrator_node_id(str(uuid.uuid5(CANONICAL_NAMESPACE, key)))


def chain_node_id(source_id: str, index: int = 0) -> str:
    """Canonical Chain graph node id (``chn:<source_id>-<index>``)."""
    if source_id.startswith(CHAIN_ID_PREFIX):
        return source_id
    return f"{CHAIN_ID_PREFIX}{bare_source_id(source_id)}-{index}"


def grading_node_id(source_id: str) -> str:
    """Canonical Grading graph node id (``grd:<source_id>``)."""
    return f"{GRADING_ID_PREFIX}{bare_source_id(source_id)}"


def is_double_prefixed(node_id: str) -> bool:
    """True if *node_id* carries a doubled leading corpus (the main#139 bug).

    Strict detector for tests / CI — unlike :func:`bare_source_id` it does not
    repair, it only reports, so a guard can fail loudly on a producer that
    emits doubled ids.
    """
    body = node_id
    for prefix in _TYPE_PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix) :]
            break
    segments = body.split(ID_DELIMITER)
    return len(segments) >= 2 and segments[0] in SOURCE_CORPORA and segments[0] == segments[1]


def validate_source_id(source_id: str) -> list[str]:
    """Return a list of contract violations for *source_id* (empty == valid).

    Checks the grammar: a known leading corpus, at least a collection segment,
    no empty segments, and no doubled leading corpus. Used by the builder and by
    tests to assert collision-safety across all 8 sources.
    """
    problems: list[str] = []
    if not source_id:
        return ["source_id is empty"]
    segments = source_id.split(ID_DELIMITER)
    if segments[0] not in SOURCE_CORPORA:
        problems.append(
            f"leading segment {segments[0]!r} is not a known corpus "
            f"(expected one of {sorted(SOURCE_CORPORA)})"
        )
    if len(segments) < 2:
        problems.append("source_id has no collection segment")
    if any(seg == "" for seg in segments):
        problems.append("source_id has an empty segment")
    if is_double_prefixed(source_id):
        problems.append(f"source_id has a doubled leading corpus: {source_id!r}")
    return problems
