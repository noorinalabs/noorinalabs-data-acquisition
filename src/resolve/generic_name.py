"""Detect over-collapsing *generic* narrator names (da#337 same-name split).

The canonical narrator id is a pure function of the normalized name
(:func:`src.parse.identity.make_canonical_id`). That is the cross-source collapse
da#99 wants, but it also **over-merges**: a name that is too generic to identify a
single person — a bare kunya (``أبو عبد الله``, "father of ʿAbd Allāh"), a
single-token mononym (``سفيان``), a ``بن``/``ابن`` patronymic with no ism, a bare
two-token fragment — collapses many *distinct* people onto one ``nar:`` node. Such
a node then accretes every source's mentions and reports a wildly inflated
betweenness centrality (a spurious "hub").

:func:`is_generic_name` is the pure eligibility screen for the da#337 split stage
(PR-2): it flags the names whose node is a same-name collapse worth splitting,
without deciding *how* to split. It is deliberately conservative on two axes so it
never nominates a name that safely denotes one person:

* **Thinness** — a full ``ism + nasab`` such as ``محمد بن اسماعيل البخاري`` carries
  ≥3 significant (non-connector) tokens and is specific; only ≤2-significant-token
  names are eligible.
* **Aggregation** — a generic name only distorts centrality once its node has
  accreted many mentions, so a mention floor (:data:`GENERIC_MIN_MENTIONS`) gates
  eligibility; a rare bare kunya with a handful of mentions is left alone.

Reuses the shared genealogical-connector vocabulary and tokenizer from
:mod:`src.resolve.fuzzy_cluster` (``_CONNECTOR_TOKENS`` / ``_significant_tokens``)
so "significant token" means exactly what it means to the fuzzy-clustering pass.
Pure, deterministic, no IO — the input is an already-normalized name
(``src.utils.arabic.normalize_arabic`` output).
"""

from __future__ import annotations

from src.resolve.fuzzy_cluster import _CONNECTOR_TOKENS, _significant_tokens
from src.utils.arabic import normalize_arabic

__all__ = ["GENERIC_MIN_MENTIONS", "is_generic_name"]

# Mention floor for the aggregation gate: a generic name is only a centrality
# hazard once its collapsed node has accreted at least this many mentions. Below
# it, an over-merged generic node is too small to distort the graph, so the split
# stage leaves it alone (precision-first, mirroring the da#248 mononym guards).
GENERIC_MIN_MENTIONS = 50

# Max significant (non-connector) tokens for a name to count as "thin". A full
# ism+nasab (``محمد بن اسماعيل البخاري`` → {محمد, اسماعيل, البخاري}, 3 significant
# tokens) is specific enough NOT to over-merge; every generic shape carries ≤2.
_MAX_THIN_SIGNIFICANT_TOKENS = 2

# Kunya lead particles (أبو/أبي/أم) and nasab connectors (ابن/بن), normalized the
# SAME way an input name is so a raw-token comparison matches. Both are subsets of
# the shared ``_CONNECTOR_TOKENS`` genealogical vocabulary — the asserts fail loud
# if a normalization change ever drifts these literals out of that set.
_KUNYA_MARKERS = frozenset(normalize_arabic(t) for t in ("أبو", "أبي", "أم"))
_NASAB_MARKERS = frozenset(normalize_arabic(t) for t in ("ابن", "بن"))
assert _KUNYA_MARKERS <= _CONNECTOR_TOKENS
assert _NASAB_MARKERS <= _CONNECTOR_TOKENS


def is_generic_name(name_ar_normalized: str, mention_count: int) -> bool:
    """True when *name_ar_normalized* is a generic, over-collapsing narrator name.

    Returns ``True`` iff ALL hold:

    1. **Thin name** — ≤ :data:`_MAX_THIN_SIGNIFICANT_TOKENS` significant
       (non-connector) tokens, so a full ``ism + nasab`` (≥3) is never flagged.
    2. **Generic shape** — at least one of: a bare kunya (leads with أبو/أبي/أم and
       carries no ``بن``/``ابن`` nasab); a bare nasab (leads with ابن/بن and a single
       following token, i.e. no leading ism); a single-token mononym; or a
       ≤2-token fragment.
    3. **Aggregation gate** — ``mention_count >= GENERIC_MIN_MENTIONS``.

    Pure and deterministic. The input must be pre-normalized
    (``src.utils.arabic.normalize_arabic``). Empty/blank names return ``False``
    (never raises).
    """
    name = name_ar_normalized
    if not name or not name.strip():
        return False

    # 3. Aggregation gate — a small over-merged node is not a centrality hazard.
    if mention_count < GENERIC_MIN_MENTIONS:
        return False

    # 1. Thin name — a full ism+nasab (≥3 significant tokens) is specific enough.
    if len(_significant_tokens([name])) > _MAX_THIN_SIGNIFICANT_TOKENS:
        return False

    # 2. Generic shape — the enumerated same-name-collapse forms (da#337 AC). The
    # shapes are listed explicitly (rather than collapsed to "≤2 raw tokens") so
    # each maps to an acceptance criterion; ``bare_kunya`` is the one that reaches
    # past a 2-token name (a kunya-led 3-token name with no disambiguating nasab).
    raw_tokens = name.split()
    first = raw_tokens[0]
    has_nasab = any(tok in _NASAB_MARKERS for tok in raw_tokens)

    bare_kunya = first in _KUNYA_MARKERS and not has_nasab
    bare_nasab = first in _NASAB_MARKERS and len(raw_tokens) == 2
    single_mononym = len(raw_tokens) == 1
    short_fragment = len(raw_tokens) <= 2

    return bare_kunya or bare_nasab or single_mononym or short_fragment
