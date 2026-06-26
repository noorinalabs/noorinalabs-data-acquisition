"""Geographic normalization + travel-plausibility for narrator disambiguation.

The disambiguation pipeline's geographic stage (``disambiguate._geographic_filter``)
needs two things the raw bio data does not provide directly:

1. **Location normalization** — narrator ``birth_location`` / ``death_location`` are
   free-text strings with inconsistent transliteration across corpora ("Kufa",
   "al-Kufah", "الكوفة"). This module maps the common historical Islamic scholarly
   centers to a canonical region so two spellings of the same place compare equal.
2. **A travel-plausibility relation** — could a single narrator plausibly bridge two
   chain neighbours given where each was based? Modelled coarsely as a west→east
   longitudinal *zone* per region: two regions are travel-plausible when they are
   the same, when either is the pilgrimage hub (Hijaz — connected to the whole
   Islamic world by hajj), or when their zones are within a conservative threshold.

**Conservative by construction (da#139).** The stage that consumes this must never
silently drop a *valid* match on noisy or missing location data, so every "I don't
know" answer here resolves toward *plausible*:

* an un-recognised free-text string → :func:`resolve_region` returns ``None`` (the
  caller keeps the match — soft constraint, mirroring the temporal filter's handling
  of a missing death year);
* an unknown region in :func:`regions_plausible` → ``True``.

Only a confidently-recognised pair of *far-apart* regions (e.g. Andalus ↔ Khurasan)
is judged implausible. This filters the worst homonym confusions — an Andalusian
scholar can't be the missing link between two Transoxianan narrators — while leaving
the well-connected core of the Islamic world untouched.

The canonical seed below is intentionally modest: only well-established centres with
unambiguous identity are included. It is *reference* data, not a guess — adding the
full historical gazetteer is future enrichment work (a populated ``locations.yaml`` /
``Location`` ontology, see ``src/models/historical.py``), and :func:`resolve_region`
will transparently benefit from richer aliases without any change to the filter.
"""

from __future__ import annotations

from src.utils.arabic import is_arabic, normalize_arabic

__all__ = [
    "is_travel_plausible",
    "regions_plausible",
    "resolve_region",
]


# ---------------------------------------------------------------------------
# Region travel model — coarse west→east longitudinal zones.
# ---------------------------------------------------------------------------
# Lower = further west. Used only for a distance-style plausibility check; the
# absolute values are not meaningful, only their pairwise gaps.
_REGION_ZONE: dict[str, int] = {
    "andalus": 0,
    "maghrib": 1,
    "egypt": 2,
    "sham": 3,
    "hijaz": 3,
    "yemen": 3,
    "iraq": 4,
    "jibal": 5,
    "fars": 5,
    "khurasan": 6,
    "transoxiana": 7,
}

# Regions connected to the whole Islamic world by the pilgrimage — plausible with
# every other region regardless of zone distance.
_HUB_REGIONS: frozenset[str] = frozenset({"hijaz"})

# Maximum zone gap still considered a plausible single-narrator span. A gap of 4
# keeps the long-but-attested rihla corridors (e.g. Andalus↔Iraq, Iraq↔Transoxiana)
# plausible while rejecting antipodal pairs (Andalus↔Khurasan, Egypt↔Transoxiana).
_TRAVEL_ZONE_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Canonical location seed: alias forms -> region.
# ---------------------------------------------------------------------------
# Each entry maps a region to the free-text forms (English / transliteration /
# Arabic) that denote a centre in that region, plus the region's own name forms.
# Arabic forms are stored raw and normalised at module load (see _ALIAS_TO_REGION).
_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "hijaz": (
        "hijaz",
        "al-hijaz",
        "الحجاز",
        "mecca",
        "makka",
        "makkah",
        "bakkah",
        "مكة",
        "medina",
        "madina",
        "madinah",
        "al-madina",
        "al-madinah",
        "yathrib",
        "المدينة",
        "المدينه",
        "taif",
        "al-taif",
        "الطائف",
    ),
    "iraq": (
        "iraq",
        "al-iraq",
        "العراق",
        "kufa",
        "kufah",
        "al-kufah",
        "الكوفة",
        "الكوفه",
        "basra",
        "basrah",
        "al-basrah",
        "البصرة",
        "البصره",
        "baghdad",
        "madinat al-salam",
        "بغداد",
        "wasit",
        "واسط",
        "mosul",
        "mawsil",
        "al-mawsil",
        "الموصل",
    ),
    "sham": (
        "sham",
        "al-sham",
        "syria",
        "الشام",
        "damascus",
        "dimashq",
        "دمشق",
        "homs",
        "hims",
        "حمص",
        "aleppo",
        "halab",
        "حلب",
        "jerusalem",
        "al-quds",
        "bayt al-maqdis",
        "القدس",
    ),
    "egypt": (
        "egypt",
        "misr",
        "مصر",
        "fustat",
        "al-fustat",
        "الفسطاط",
        "cairo",
        "al-qahira",
        "القاهرة",
        "alexandria",
        "iskandariyya",
        "al-iskandariyya",
        "الإسكندرية",
        "الاسكندرية",
    ),
    "yemen": (
        "yemen",
        "al-yaman",
        "اليمن",
        "sanaa",
        "sana",
        "صنعاء",
        "aden",
        "عدن",
    ),
    "khurasan": (
        "khurasan",
        "khorasan",
        "خراسان",
        "nishapur",
        "naysabur",
        "نيسابور",
        "merv",
        "marw",
        "مرو",
        "herat",
        "harat",
        "هراة",
        "balkh",
        "بلخ",
    ),
    "transoxiana": (
        "transoxiana",
        "ma wara al-nahr",
        "ما وراء النهر",
        "bukhara",
        "بخارى",
        "samarqand",
        "samarkand",
        "سمرقند",
    ),
    "jibal": (
        "jibal",
        "al-jibal",
        "الجبال",
        "rayy",
        "al-rayy",
        "الري",
        "isfahan",
        "isbahan",
        "أصفهان",
        "اصفهان",
        "hamadan",
        "hamadhan",
        "همذان",
        "qom",
        "qum",
        "قم",
    ),
    "fars": (
        "fars",
        "persia",
        "فارس",
        "shiraz",
        "شيراز",
    ),
    "maghrib": (
        "maghrib",
        "al-maghrib",
        "المغرب",
        "qayrawan",
        "kairouan",
        "al-qayrawan",
        "القيروان",
        "fes",
        "fez",
        "فاس",
    ),
    "andalus": (
        "andalus",
        "al-andalus",
        "الأندلس",
        "الاندلس",
        "cordoba",
        "qurtuba",
        "قرطبة",
        "seville",
        "ishbiliyya",
        "إشبيلية",
    ),
}


def _norm_alias(text: str) -> str:
    """Normalise one location token to a lookup key.

    Arabic strings go through the full Arabic pipeline; Latin transliterations are
    lower-cased, whitespace-collapsed, and stripped of a leading ``al-``/``al`` so
    "al-Kufah" and "Kufah" collapse together.
    """
    text = text.strip()
    if not text:
        return ""
    if is_arabic(text):
        return normalize_arabic(text)
    text = " ".join(text.lower().split())
    for article in ("al-", "al ", "el-", "el "):
        if text.startswith(article):
            text = text[len(article) :]
            break
    return text


# alias-key -> region, built once at import.
_ALIAS_TO_REGION: dict[str, str] = {}
for _region, _aliases in _REGION_ALIASES.items():
    for _alias in _aliases:
        _key = _norm_alias(_alias)
        if _key:
            _ALIAS_TO_REGION.setdefault(_key, _region)


def resolve_region(location: str | None) -> str | None:
    """Resolve a free-text location string to a canonical region, or ``None``.

    Tries a whole-string match first, then falls back to per-token matching so a
    descriptive form ("المدينة المنورة", "Basra, Iraq") still resolves on its
    recognisable token. Returns ``None`` for anything unrecognised — the caller
    must treat ``None`` as "no geographic signal" and keep the match.
    """
    if not location:
        return None
    whole = _norm_alias(location)
    if whole in _ALIAS_TO_REGION:
        return _ALIAS_TO_REGION[whole]
    # Token fallback: split on whitespace and common separators.
    for raw_token in whole.replace(",", " ").replace("/", " ").split():
        token = _norm_alias(raw_token)
        if token in _ALIAS_TO_REGION:
            return _ALIAS_TO_REGION[token]
    return None


def regions_plausible(region_a: str | None, region_b: str | None) -> bool:
    """Could a narrator plausibly be based across *region_a* and *region_b*?

    Conservative: returns ``True`` whenever either region is unknown or a hub, and
    only ``False`` for two recognised regions whose travel zones are further apart
    than :data:`_TRAVEL_ZONE_THRESHOLD`.
    """
    if region_a is None or region_b is None:
        return True
    if region_a == region_b:
        return True
    if region_a in _HUB_REGIONS or region_b in _HUB_REGIONS:
        return True
    zone_a = _REGION_ZONE.get(region_a)
    zone_b = _REGION_ZONE.get(region_b)
    if zone_a is None or zone_b is None:
        return True
    return abs(zone_a - zone_b) <= _TRAVEL_ZONE_THRESHOLD


def is_travel_plausible(location_a: str | None, location_b: str | None) -> bool:
    """Free-text convenience wrapper over :func:`regions_plausible`.

    Resolves both strings to regions first; an unresolvable string yields ``None``
    and therefore a plausible (kept) verdict.
    """
    return regions_plausible(resolve_region(location_a), resolve_region(location_b))
