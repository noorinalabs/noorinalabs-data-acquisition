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
  so NO ``mis`` Hadith nodes are loaded (``network_edges_mis`` still loads as
  narrator-transmission edges).
* every other source — all collections (value ``None``).

This is the **per-source** half of the dedup and is the deterministic part that
MUST hold before the production load. Cross-edition canonical-identity dedup
(same hadith by collection+number across editions) is a tracked fast-follow.

This module is the single source of truth, consumed by the graph node loader
(``src.graph.load_nodes``) so a fresh ``run_all`` (the path production uses)
produces the deduped graph without manual surgery.
"""

from __future__ import annotations

# source_corpus -> allowed collection slugs.
#   None           = load all collections for this source. Equivalent to the
#                    default for any source not listed here, so an explicit
#                    ``"x": None`` entry and an absent key behave identically.
#   frozenset(...) = load only these collections.
#   frozenset()    = load NO Hadith nodes for this source (e.g. ``mis``: its
#                    chains/edges still load via the edge loader).
HADITH_COMPOSITION: dict[str, frozenset[str] | None] = {
    "halimbahae": frozenset({"musnad_ahmad_ibn-hanbal", "sunan_al-darimi", "maliks_muwataa"}),
    "fawaz": frozenset({"nawawi", "dehlawi", "qudsi"}),
    "mis": frozenset(),
    # open_hadith is dropped at the registry (active=False); listed here as a
    # defence-in-depth no-op in case its parquet is ever present in a staging dir.
    "open_hadith": frozenset(),
}


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
