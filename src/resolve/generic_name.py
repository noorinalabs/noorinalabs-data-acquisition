"""Detect over-collapsing *generic* narrator names (da#337 same-name split).

The canonical narrator id is a pure function of the normalized name
(:func:`src.parse.identity.make_canonical_id`). That is the cross-source collapse
da#99 wants, but it also **over-merges**: a name that is too generic to identify a
single person — a bare kunya (``أبو عبد الله``, "father of ʿAbd Allāh"), a
single-token mononym (``سفيان``), a ``بن``/``ابن`` patronymic with no ism, a bare
two-token fragment — collapses many *distinct* people onto one ``nar:`` node. Such
a node then accretes every source's mentions and reports a wildly inflated
betweenness centrality (a spurious "hub").

:func:`is_generic_name` is a **recall-first candidacy screen** for the da#337 split
stage (PR-2): it over-includes the names whose node *might* be a same-name collapse,
without deciding *how* — or *whether* — to split. It deliberately flags every thin
(≤2-significant-token) name above a mention floor, **including a specific
``ism + nisba`` such as ``سفيان الثوري`` (Sufyān al-Thawrī)**. That is not a bug:
distinguishing a disambiguating nisba from a theophoric compound is exactly the
multi-referent judgement that needs downstream evidence, not a name-shape heuristic
— a naive "second token is ``ال``-prefixed ⇒ nisba ⇒ safe" rule would wrongly
protect ``عبد الله`` (``الله`` is ``ال``-prefixed too), which we DO want screened in.

**The split decision is gated downstream in PR-2 and MUST NEVER be made on this
screen alone.** PR-2 splits only on independent multi-referent evidence — ≥2
well-separated, well-supported death-year bands — so a genuinely single person like
``سفيان الثوري`` clusters to one band and PR-2 *abstains*, never fragmenting it.
This screen trades precision for recall on purpose; the precision lives in that
evidence gate.

Two shape axes keep the candidacy set to plausibly-generic names:

* **Thinness** — a full ``ism + nasab`` such as ``محمد بن اسماعيل البخاري`` carries
  ≥3 significant (non-connector) tokens and is screened OUT; only names with ≤2
  significant tokens are candidates.
* **Aggregation** — a mention floor (:data:`GENERIC_MIN_MENTIONS`) drops nodes too
  small to distort centrality even if over-merged. (It is not a precision guard —
  a real over-collapsed hub has thousands of mentions; that is what PR-2 is for.)

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

# Mention floor for the aggregation gate: below this many mentions an over-merged
# generic node is too small to distort centrality, so the split stage leaves it
# alone. This is a size cut-off, NOT a precision guard — a real over-collapsed hub
# carries thousands of mentions and still screens in; PR-2's evidence gate is what
# keeps a genuinely single person from being split.
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
    """Screen *name_ar_normalized* IN as a da#337 split *candidate* (recall-first).

    Returns ``True`` iff ALL hold:

    1. **Thin name** — ≤ :data:`_MAX_THIN_SIGNIFICANT_TOKENS` significant
       (non-connector) tokens, so a full ``ism + nasab`` (≥3, e.g.
       ``محمد بن اسماعيل البخاري``) is screened OUT.
    2. **Generic shape** — a ≤2-token name (mononym, a bare ``بن``/``ابن`` patronymic,
       or ANY two-token compound — including an ``ism + nisba`` like
       ``سفيان الثوري``, screened in by design; see the module note), OR a kunya-led
       name (leads with أبو/أبي/أم) carrying no ``بن``/``ابن`` nasab — the only shape
       that reaches past two tokens (e.g. ``أبو عبد الله``).
    3. **Aggregation gate** — ``mention_count >= GENERIC_MIN_MENTIONS``.

    Recall-first: this over-includes 2-token names on purpose and NEVER decides a
    split — that is gated downstream (da#337 PR-2) on independent multi-referent
    evidence (distinct death-year bands), which abstains on a genuinely single
    person such as ``سفيان الثوري``. Pure and deterministic; the input must be
    pre-normalized (``src.utils.arabic.normalize_arabic``); empty/blank names
    return ``False`` (never raises).
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

    # 2. Generic shape — two load-bearing forms. ``short_fragment`` screens in every
    # ≤2-token name (mononym, bare بن/ابن patronymic, and any two-token compound —
    # INCLUDING an ism+nisba like سفيان الثوري, deliberately: this is a candidacy
    # screen, and PR-2's death-year-band evidence gate, not this function, decides
    # the actual (non-)split). ``kunya_led`` is the ONLY form reaching past two
    # tokens: a kunya (أبو/أبي/أم) with no disambiguating بن/ابن nasab (e.g.
    # ``أبو عبد الله``, 3 raw tokens).
    raw_tokens = name.split()
    short_fragment = len(raw_tokens) <= 2
    kunya_led = raw_tokens[0] in _KUNYA_MARKERS and not any(
        tok in _NASAB_MARKERS for tok in raw_tokens
    )
    return short_fragment or kunya_led
