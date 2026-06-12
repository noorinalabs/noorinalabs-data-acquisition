"""Derive a narrator-level :class:`SectAffiliation` from observed source corpora.

Narrator sect is *not* the collection-level binary :class:`~src.models.enums.Sect`
(``sunni``/``shia``). A narrator — especially a female *muhaddithat* — can transmit
in chains of more than one tradition, so the narrator-level determination is
:class:`~src.models.enums.SectAffiliation` (``sunni``/``shia``/``neutral``/``unknown``)
and it is made at **resolution time** by aggregating the corpora a canonical
narrator was actually seen in (da#103, epic #81).

Aggregation rule (``derive_sect_affiliation``):

* observed only in Sunni-tradition corpora  → ``sunni``
* observed only in Shia-tradition corpora   → ``shia``
* observed in **both** traditions           → ``neutral`` (transmits across them)
* no corpus maps to a known tradition        → ``unknown``

The corpus→tradition map below is keyed by :class:`~src.models.enums.SourceCorpus`
*values*, mirroring the per-corpus ``SECT`` the parsers stamp onto Hadith/Collection
nodes (``src/parse/sunnah_api.py``, ``thaqalayn.py``, ``lk_corpus.py``, …). Two
corpora are intentionally left unmapped: ``fawaz`` (sect is per-collection, not
per-corpus) and ``itqan`` (a mixed rijal database) — both contribute ``unknown``
rather than a guess.

``normalize_corpus`` reconciles the **bio provenance inconsistency** #103 flags:
``narrators_bio_*`` rows carry a free-text ``source`` that is not always a
``SourceCorpus`` value (e.g. the sanadset narrator bios use
``source="kaggle_narrators"``). Producers normalize through it so the scalar
``Narrator.source_corpus`` stays inside the enum domain (the ``write_parquet``
provenance gate enforces this), while genuinely-unknown provenance maps to
``None`` rather than leaking a non-enum string onto the node.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.models.enums import Sect, SectAffiliation, SourceCorpus

__all__ = [
    "derive_sect_affiliation",
    "normalize_corpus",
    "primary_corpus",
    "SECT_BY_CORPUS",
]

_VALID_CORPORA: frozenset[str] = frozenset(c.value for c in SourceCorpus)

# Free-text bio ``source`` labels that are not yet ``SourceCorpus`` values (#103).
_CORPUS_ALIASES: dict[str, str] = {
    "kaggle_narrators": SourceCorpus.SANADSET.value,  # sanadset narrator biographies
}

# Corpus → collection-level tradition. Unmapped corpora (``fawaz`` per-collection,
# ``itqan`` mixed rijal DB) contribute nothing → ``unknown`` when nothing maps.
SECT_BY_CORPUS: dict[str, Sect] = {
    SourceCorpus.SUNNAH.value: Sect.SUNNI,
    SourceCorpus.OPEN_HADITH.value: Sect.SUNNI,
    SourceCorpus.LK.value: Sect.SUNNI,
    SourceCorpus.SANADSET.value: Sect.SUNNI,
    SourceCorpus.MUHADDITHAT.value: Sect.SUNNI,
    SourceCorpus.THAQALAYN.value: Sect.SHIA,
}


def normalize_corpus(source: str | None) -> str | None:
    """Map a raw corpus / bio-``source`` label to a :class:`SourceCorpus` value.

    Returns the canonical corpus value, or ``None`` when *source* is empty or not
    a recognized corpus (so callers can drop it rather than tag a node with an
    out-of-domain provenance string).
    """
    if not source:
        return None
    resolved = _CORPUS_ALIASES.get(source, source)
    return resolved if resolved in _VALID_CORPORA else None


def primary_corpus(corpora: Iterable[str]) -> str | None:
    """Return a deterministic primary corpus for a multi-source narrator.

    The canonical narrator carries the full set in ``source_corpora``; the scalar
    ``source_corpus`` (mirroring the Hadith/Collection node property) is the
    lexicographically-first observed corpus, or ``None`` when none were observed.
    Input is assumed already normalized; falsy entries are ignored defensively.
    """
    uniq = sorted({c for c in corpora if c})
    return uniq[0] if uniq else None


def derive_sect_affiliation(corpora: Iterable[str]) -> str:
    """Derive a :class:`SectAffiliation` value from the corpora a narrator spans.

    See the module docstring for the aggregation rule. Each corpus is normalized
    first (so a not-yet-enum-aligned bio ``source`` still resolves), then mapped to
    its tradition. Returns the enum *value* (a plain ``str``) so it drops straight
    into a Parquet string column and the Neo4j ``Narrator.sect_affiliation``
    property.
    """
    sects = set()
    for raw in corpora:
        corpus = normalize_corpus(raw)
        if corpus is not None and corpus in SECT_BY_CORPUS:
            sects.add(SECT_BY_CORPUS[corpus])
    if not sects:
        return SectAffiliation.UNKNOWN.value
    if sects == {Sect.SUNNI}:
        return SectAffiliation.SUNNI.value
    if sects == {Sect.SHIA}:
        return SectAffiliation.SHIA.value
    return SectAffiliation.NEUTRAL.value
