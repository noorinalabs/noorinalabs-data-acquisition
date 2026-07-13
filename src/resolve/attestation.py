"""Derive a narrator-level :class:`Attestation` tag from its isnad mention count (da#370).

A canonical narrator is ``biographical_only`` when it carries **no** isnad
transmission mention — a record promoted from a rijal / biographical source
(itqan / kaggle / muhaddithat) rather than seen transmitting in a chain. The signal
is already present on the canonical record: ``bio_promote`` stamps ``mention_count = 0``
on a bio-promoted narrator ("a bio is provenance, not a chain hit"), while a narrator
disambiguated from real chain mentions carries ``mention_count >= 1``.

Why this is derived at every canonical-table build site, not once
----------------------------------------------------------------
``narrators_canonical.parquet`` has more than one producer that (re)computes
``mention_count``: ``disambiguate`` mints rows from mentions, ``bio_promote`` MERGEs
bio rows in (``mention_count = 0``), ``fuzzy_cluster`` collapses ids and **sums** the
survivors' ``mention_count``, and ``narrator_split`` (da#337) reduces a generic
primary's count and mints peeled band rows. A tag computed once and carried across
those steps would go stale exactly when one of them changes ``mention_count`` (a
bio-only record folded into an attested one; a primary peeled down to zero). So —
mirroring :func:`derive_sect_affiliation`, which is likewise re-derived from the
merged ``source_corpora`` at each of those sites — the attestation is re-derived from
the row's **final** ``mention_count`` at every one of those four sites. It is a pure
function of that one field, so re-deriving is always consistent and never depends on
producer ordering.

The remaining canonical writers in ``RESOLVE_STEP_ORDER`` — ``date_reconcile`` and
``tabaqa_dates`` — carry the tag through untouched, and that is safe **only because
they never recompute ``mention_count``**: they round-trip the schema and edit date
columns, so a value an earlier site derived stays valid. A future step that mutates
``mention_count`` MUST add a re-derivation, exactly like the four above.

Note the tag is about *isnad* attestation, not graph degree: a ``biographical_only``
narrator whose name later resolves to a ``STUDIED_UNDER`` network endpoint at load
time still has no isnad transmission, so the tag stands. It is not the same predicate
as "zero-degree" — that is why ``orphan_narrators.cypher`` excludes bio-only nodes
explicitly rather than inferring bio-only from degree.
"""

from __future__ import annotations

from src.models.enums import Attestation


def derive_attestation(mention_count: object) -> str:
    """Return the :class:`Attestation` value for a canonical narrator.

    ``isnad_attested`` when the narrator carries a positive isnad ``mention_count``,
    else ``biographical_only`` (a bio-promoted record with ``mention_count`` 0 or
    ``None``). Never returns ``unknown``: a canonical narrator always has a known
    ``mention_count``. Returns the enum *value* (a ``str``) so it drops straight into
    the ``NARRATORS_CANONICAL_SCHEMA`` string column.

    Accepts ``object`` because the canonical-row dicts in the resolve layer type this
    field loosely; only a positive ``int`` counts as attested (mirrors the
    ``mc if isinstance(mc, int)`` guard the fuzzy-cluster merge already uses).
    """
    if isinstance(mention_count, int) and mention_count > 0:
        return Attestation.ISNAD_ATTESTED.value
    return Attestation.BIOGRAPHICAL_ONLY.value
