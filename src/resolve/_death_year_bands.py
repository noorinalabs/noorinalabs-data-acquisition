"""Shared resolve-stage death-year sanity bands (da#464).

``fuzzy_cluster`` (da#380) and ``narrator_unify`` (da#431) each veto a
merge/unify decision on a death-year conflict, weighed at two bands:

* :data:`DEATH_YEAR_TOLERANCE` — the tight band. Two *trusted* years
  (corroborated, or an untagged/legacy record) disagreeing by more than this
  many AH years block the merge outright; one source rounding a year by a
  year or two is fine, a whole generation apart is not.
* :data:`GROSS_DEATH_SPREAD` — the loose band. A much wider sanity check that
  still catches a distinct namesake even when neither year carries full
  trust — an ``uncorroborated`` fuzzy-bio year in ``fuzzy_cluster``, or a
  curated unify-group member in ``narrator_unify``.

Both stages must weigh an uncorroborated/noisy year the same way — that
cross-stage invariant used to be two independent literals a code comment
merely *promised* stayed equal (one retune away from silent drift). Hoisting
both constants here makes the invariant structural: there is only one place
to change either band, and both stages import it.
"""

from __future__ import annotations

# Two *trusted* death years (corroborated, or untagged/legacy) disagreeing by
# more than this many AH years block a merge/unify outright.
DEATH_YEAR_TOLERANCE = 2

# Loose sanity band (in AH years) applied when at least one death year is not
# fully trusted (fuzzy_cluster's `uncorroborated` tag; narrator_unify's
# curated-but-noisy group members). Wide enough to pass a genuine
# generation-default noise gap, tight enough to still refuse a
# cross-generation namesake.
GROSS_DEATH_SPREAD = 50
