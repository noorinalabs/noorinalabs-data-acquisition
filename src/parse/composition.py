"""Canonical corpus composition — the owner-confirmed dedup, encoded (da#191).

Several acquired corpora overlap. The pre-cutover corpus-integrity audit found:

* ``open_hadith`` is a 100% duplicate of ``halimbahae`` (handled in the registry:
  ``open_hadith`` is marked ``active=False`` so it is never acquired/parsed);
* the six canonical Sunni books (Bukhari, Muslim, Abu Dawud, Tirmidhi, Nasa'i,
  Ibn Majah) are loaded by ``lk`` AND ``halimbahae`` AND ``fawaz``, and Sahih
  Muslim again by ``mis``.

Loading every copy would double-count hadith on production. The owner-confirmed
composition (relayed via the Program Director) keeps each book once:

* ``lk`` — the six-books canonical spine (richest metadata: separated
  ``matn_ar`` + ``matn_en`` + ``grade``). Loads all its collections.
* ``halimbahae`` — its UNIQUE books only (Musnad Ahmad, Darimi, Malik).
* ``fawaz`` — its UNIQUE collections only (Nawawi, Dehlawi, Qudsi).
* ``mis`` — its multi-isnad CHAINS only. Its Sahih Muslim matn duplicates ``lk``,
  so NO ``mis`` Hadith nodes are loaded. Its transmission edges are staged as
  ``network_edges_mis.parquet`` and read by the STUDIED_UNDER glob but skipped
  row-by-row (they declare TRANSMITTED_TO, not STUDIED_UNDER), so no mis edge is
  persisted (produced, not loaded as a graph edge), pending a name-reconciliation
  crosswalk (mis Latin-transliteration ids ↔ Arabic-normalized canonical keys);
  tracked as a separate follow-up to da#364.
* every other source — all collections (value ``None``).

This is the **per-source** half of the dedup and is the deterministic part that
MUST hold before the production load.

Cross-edition canonical-identity dedup (da#220 / Path B-B2)
-----------------------------------------------------------
The per-source map above keeps each *book* once, but it cannot stop a hadith in
one corpus from duplicating the *same canonical tradition* already loaded — with
richer metadata — from another corpus. That gap matters for ``sanadset``: B1
(da#219) re-links its 650k hadiths to their collections, but ``sanadset`` is a
raw re-publication that overlaps the curated spine (``lk`` / ``halimbahae`` /
``fawaz`` …). Admitting all of it without a cross-edition check would
**double-count** the canonical books on production.

:func:`canonical_matn_identity` mints a deterministic identity for a hadith from
its **normalized matn** — the one signal both editions actually share. (The
collection slug does NOT survive across editions: ``sanadset`` keys collections
by numeric ``book_id``, the curated sources by name, so a literal
"collection+number" key never collides between them. The normalized matn does.)
The graph node loader (``src.graph.load_nodes._load_hadiths``) builds the set of
identities occupied by the **curated** sources (everything NOT in
:data:`CROSS_EDITION_DEDUP_SOURCES`) and drops any hadith from a dedup-source
whose identity collides — **curated wins**, because it carries the richer,
deduplicated metadata. The match is *exact normalized-matn equality*: it only
ever removes a genuine textual duplicate, never a distinct tradition (the safe,
irreversible-load-friendly direction). Sub-exact near-duplicates are a *linking*
concern, not a dedup one, and remain the job of ``PARALLEL_OF`` detection
(``src.resolve.parallels``).

This module is the single source of truth, consumed by the graph node loader
(``src.graph.load_nodes``) so a fresh ``run_all`` (the path production uses)
produces the deduped graph without manual surgery.
"""

from __future__ import annotations

import hashlib

from src.parse.identity import HADITH_ID_PREFIX, ID_DELIMITER
from src.utils.arabic import is_arabic, normalize_arabic_uncached, strip_to_letters

# source_corpus -> allowed collection slugs.
#   None           = load all collections for this source. Equivalent to the
#                    default for any source not listed here, so an explicit
#                    ``"x": None`` entry and an absent key behave identically.
#   frozenset(...) = load only these collections.
#   frozenset()    = load NO Hadith nodes for this source (e.g. ``mis``: its
#                    transmission edges are staged as ``network_edges_mis.parquet``
#                    and read by the STUDIED_UNDER glob but skipped (they declare
#                    TRANSMITTED_TO), so no mis edge is persisted — produced, not
#                    loaded as a graph edge — pending a name-reconciliation
#                    crosswalk; da#364).
HADITH_COMPOSITION: dict[str, frozenset[str] | None] = {
    "halimbahae": frozenset({"musnad_ahmad_ibn-hanbal", "sunan_al-darimi", "maliks_muwataa"}),
    "fawaz": frozenset({"nawawi", "dehlawi", "qudsi"}),
    "mis": frozenset(),
    # open_hadith is dropped at the registry (active=False); listed here as a
    # defence-in-depth no-op in case its parquet is ever present in a staging dir.
    "open_hadith": frozenset(),
    # sanadset (da#219 / Path B-B1): admit ALL collections at the per-source
    # level. ``sanadset`` keeps its full collection breadth — that is the reason
    # Path B was chosen over a purge — so this is NOT tightened to a per-source
    # allowlist. The double-counting risk the ADR flags is handled per-*hadith*
    # by the cross-edition dedup (da#220 / B2): ``sanadset`` is listed in
    # :data:`CROSS_EDITION_DEDUP_SOURCES`, so a sanadset hadith whose normalized
    # matn matches an already-loaded curated tradition is dropped at load (curated
    # wins). ``None`` here is behaviorally identical to the unlisted-source
    # default; listing it keeps sanadset's admission explicit next to that gate.
    "sanadset": None,
}

# Sources whose hadiths are deduplicated against the curated spine by
# cross-edition canonical identity (da#220 / Path B-B2). A hadith from one of
# these sources is dropped at graph-load time when its
# :func:`canonical_matn_identity` collides with a hadith already admitted from a
# *curated* (non-dedup) source — "curated wins", because the curated editions
# carry the richer, deduplicated metadata. This is the per-hadith complement to
# the per-source/per-collection :data:`HADITH_COMPOSITION` gate above: the map
# keeps each *book* once; this keeps each *tradition* once across editions.
CROSS_EDITION_DEDUP_SOURCES: frozenset[str] = frozenset({"sanadset"})

# Minimum number of normalized matn tokens for a hadith to carry a reliable
# cross-edition identity. A matn shorter than this is too generic to distinguish
# traditions (e.g. a bare formula), so such a hadith is never deduped — it is
# always kept, the conservative direction for an irreversible production load.
_MIN_IDENTITY_TOKENS = 3


def is_canonical_hadith(source_corpus: str, collection_name: str) -> bool:
    """Return True if a ``(source_corpus, collection_name)`` Hadith belongs in the
    canonical (production) graph, per :data:`HADITH_COMPOSITION`.

    A source not present in the composition map — or present with an explicit
    ``None`` value — loads all of its collections (da#196). Only a non-empty
    ``frozenset`` restricts to an allowlist, and an empty ``frozenset`` drops
    all Hadith nodes for that source.
    """
    if source_corpus not in HADITH_COMPOSITION:
        return True
    allowed = HADITH_COMPOSITION[source_corpus]
    if allowed is None:
        return True
    return collection_name in allowed


def is_canonical_hadith_id(hadith_id: str) -> bool:
    """Return True if the hadith identified by *hadith_id* would be loaded as a
    canonical Hadith node, per :data:`HADITH_COMPOSITION`.

    The id-parsing complement to :func:`is_canonical_hadith` for the chain-edge
    loader (``src.graph.load_edges``). The node loader has the hadith row's
    ``source_corpus`` + ``collection_name`` columns to hand; the edge loader only
    has the narrator-mention's ``hadith_id`` (a ``source_id``, optionally
    ``hdt:``-prefixed), so it recovers ``(corpus, collection)`` from the id
    grammar ``<corpus>:<collection>[:<part>...]`` (:mod:`src.parse.identity`)
    before applying the same gate.

    Applying this on the edge path mirrors the node dedup (da#333): a
    narrator-mention whose hadith is NOT canonical — e.g. ``fawaz``'s six-books
    chains, which are deduplicated to the ``lk`` spine and so load no Hadith node
    — must not produce ``TRANSMITTED_TO`` / ``NARRATED`` edges, or it orphans an
    edge against a Hadith node that was never loaded (the ~196k
    ``fawaz:<book>:<n>`` dangling chains). An id with no recoverable collection
    segment (a bare corpus, or a toy/id-less value) defers to
    :func:`is_canonical_hadith`, which keeps any corpus not named in the
    composition map — the conservative, irreversible-load-friendly default.

    This is a **total** predicate: it strips the ``hdt:`` prefix inline rather
    than via :func:`~src.parse.identity.bare_source_id`, which raises
    :exc:`~src.parse.identity.DoubledCorpusPrefixError` on a malformed id (da#355).
    A composition gate that contracts to return a ``bool`` must not throw — and a
    malformed id is not this function's decision to make. It reports whether the
    hadith *would be* canonical; the loader's quarantine then rejects the id and
    counts the row, so the failure is recorded once, at one place, rather than
    silently dropping the edge here.
    """
    body = (
        hadith_id[len(HADITH_ID_PREFIX) :] if hadith_id.startswith(HADITH_ID_PREFIX) else hadith_id
    )
    segments = body.split(ID_DELIMITER)
    corpus = segments[0] if segments else ""
    collection = segments[1] if len(segments) > 1 else ""
    return is_canonical_hadith(corpus, collection)


def is_cross_edition_dedup_source(source_corpus: str) -> bool:
    """Return True if *source_corpus* is deduped against the curated spine (da#220).

    Hadiths from such a source are dropped at graph-load time when their
    :func:`canonical_matn_identity` collides with a tradition already admitted
    from a curated (non-dedup) source. See :data:`CROSS_EDITION_DEDUP_SOURCES`.
    """
    return source_corpus in CROSS_EDITION_DEDUP_SOURCES


def canonical_matn_identity(matn_ar: str | None, matn_en: str | None) -> str | None:
    """Deterministic cross-edition identity for a hadith, from its normalized matn.

    Two editions of the *same canonical tradition* share their matn body, so the
    normalized matn is the join key that survives across editions (unlike the
    collection slug — see the module docstring). Returns a stable hex digest of
    the normalized matn, or ``None`` when the hadith carries no matn reliable
    enough to identify it (empty, or fewer than :data:`_MIN_IDENTITY_TOKENS`
    tokens) — a ``None`` identity is never deduped, so the hadith is kept.

    The Arabic matn is preferred and normalized via
    :func:`src.utils.arabic.normalize_arabic_uncached` (diacritics / alif / hamza /
    taa marbuta / tatweel folded, whitespace collapsed), then reduced to letters and
    single spaces by :func:`src.utils.arabic.strip_to_letters`; the English matn
    (lowercased) is stripped the same way and is the fallback when no Arabic matn
    is present.

    Stripping punctuation is what makes this gate actually FIRE (da#373). Two
    editions of one tradition differ only by commas, editorial dashes and
    quotation marks, and ``normalize_arabic`` folds orthography but keeps
    punctuation — so an *exact* hash over its output almost never collided (it
    dropped 3 rows of 853,218), silently admitting every cross-edition duplicate
    it was built to remove. Punctuation is not part of the matn, so stripping it
    only ever merges genuine textual twins, never distinct traditions.
    ``src.resolve.parallels`` still tokenizes on ``normalize_arabic`` output
    alone, but it compares token *sets* by Jaccard, which tolerates the odd
    punctuation-bearing token; an exact hash cannot, which is why the strip lives
    here. The digest (not the raw matn) is returned so the loader's identity
    index stays compact for the full ~650k-row corpus.
    """
    if matn_ar and is_arabic(matn_ar):
        # Full matn bodies (matn_ar / full_text_ar) are large and mostly unique,
        # and the loader runs this over the full ~650k-row corpus, so bypass the
        # lru_cache on normalize_arabic (da#495) — caching these would retain
        # every distinct body for the process lifetime and risk OOM (memory-
        # bounding history da#337/#723). Same reasoning as src/resolve/parallels.py.
        normalized = strip_to_letters(normalize_arabic_uncached(matn_ar))
    elif matn_en and matn_en.strip():
        normalized = strip_to_letters(matn_en.lower())
    else:
        return None
    tokens = normalized.split()
    if len(tokens) < _MIN_IDENTITY_TOKENS:
        return None
    signature = " ".join(tokens)
    return hashlib.blake2b(signature.encode("utf-8"), digest_size=16).hexdigest()
