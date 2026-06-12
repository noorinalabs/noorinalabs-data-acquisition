"""Parse the Itqan rijal dataset into NARRATOR_BIO staging Parquet.

Produces one output file — ``narrators_bio_itqan.parquet`` (NARRATOR_BIO_SCHEMA)
— from the per-grade profile buckets downloaded by ``src/acquire/itqan.py``.

Each bucket file (``profiles_<grade>.json``) is a JSON object keyed by the
narrator id; every value is a profile dict. The jarh-wa-ta'dil grade is carried
both by the filename bucket and the per-profile ``grade_en`` field (they agree);
we trust the per-profile field and fall back to the bucket.

Scope (da#92a — narrators only): the ``teachers``/``students`` id lists (isnad
edges, #93) and the ``namings``/``by_name`` name variants (#94) are deliberately
NOT emitted here — this PR populates Narrator bios only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa

from src.parse.base import safe_str, write_parquet
from src.parse.schemas import NARRATOR_BIO_SCHEMA
from src.utils.arabic import normalize_arabic
from src.utils.logging import get_logger

logger = get_logger(__name__)

SOURCE = "itqan"

# Itqan grade bucket -> classical trustworthiness tier (TrustworthinessGrade).
# ``companion`` is a status rather than a jarh verdict; by the rijal consensus
# that the Companions are ``udul`` (all reliable) it maps to ``thiqa`` while the
# generation field separately records ``sahabi``.
_GRADE_TO_TRUST: dict[str, str] = {
    "reliable": "thiqa",
    "mostly_reliable": "saduq",
    "weak": "daif",
    "abandoned": "matruk",
    "fabricator": "kadhdhab",
    "companion": "thiqa",
    "unknown": "unknown",
}

# Coarse map from the Ibn-Hajar tabaqat ordinal (Arabic) to NarratorGeneration.
# Deliberately conservative: only the unambiguous ordinal words are mapped; the
# rest yield ``None`` rather than guess. The ``companion`` bucket overrides this
# to ``sahabi`` regardless of tabaqat.
_TABAQAT_GENERATION: list[tuple[str, str]] = [
    ("صحاب", "sahabi"),  # صحابي / صحابة
    ("الأولى", "sahabi"),
    ("الثانية", "tabii"),
    ("الثالثة", "tabii"),
    ("الرابعة", "tabii"),
    ("الخامسة", "taba_tabii"),
    ("السادسة", "taba_tabii"),
    ("السابعة", "taba_tabii"),
    ("الثامنة", "taba_tabii"),
    ("التاسعة", "atba_taba_tabiin"),
    ("العاشرة", "atba_taba_tabiin"),
    ("الحادية", "later"),  # الحادية عشرة (11th)
    ("الثانية عشرة", "later"),
]

# A Hijri year inside a free-text death string ("168 هـ", "بين 161 هـ إلى 170 هـ").
_YEAR_RE = re.compile(r"\d+")


def _clean(value: object) -> str | None:
    """Itqan uses ``"-"`` as a placeholder for missing fields; treat it as null."""
    s = safe_str(value)
    if s is None or s == "-":
        return None
    return s


def _extract_death_year(value: object) -> int | None:
    """Best-effort Hijri death year from a free-text death field.

    The field is prose — exact years, ranges ("بين 161 هـ إلى 170 هـ") and
    approximations ("180 هـ تقريبا"). We take the FIRST year mentioned as an
    approximate anchor; ranges/approximations are inherently lossy, so this is
    a hint for timelines, not an authoritative date.
    """
    s = _clean(value)
    if s is None:
        return None
    match = _YEAR_RE.search(s)
    if not match:
        return None
    year = int(match.group())
    # Guard against absurd values (no narrator died in year 0 or > ~1500 AH).
    if 1 <= year <= 1500:
        return year
    return None


def _generation(profile: dict[str, object], grade: str | None) -> str | None:
    """Map a profile to a NarratorGeneration value (or ``None`` if unclear)."""
    if grade == "companion":
        return "sahabi"
    tabaqat = _clean(profile.get("tabaqat"))
    if tabaqat is None:
        return None
    for needle, generation in _TABAQAT_GENERATION:
        if needle in tabaqat:
            return generation
    return None


def _parse_profile(raw_id: str, profile: dict[str, object]) -> dict[str, object | None] | None:
    """Map one Itqan profile to a NARRATOR_BIO_SCHEMA row dict."""
    name_ar = _clean(profile.get("full_name"))
    if name_ar is None:
        return None  # a bio with no name cannot key a canonical narrator

    # ``id`` is globally unique across all buckets (verified); fall back to the
    # JSON key when the field is absent.
    profile_id = safe_str(profile.get("id")) or safe_str(raw_id)
    grade = _clean(profile.get("grade_en"))
    trustworthiness = _GRADE_TO_TRUST.get(grade) if grade else None

    return {
        "bio_id": f"{SOURCE}:{profile_id}",
        "source": SOURCE,
        "name_ar": name_ar,
        "name_en": None,  # Itqan is Arabic-only — no transliteration in the source
        "name_ar_normalized": normalize_arabic(name_ar),
        "name_en_normalized": None,
        "kunya": _clean(profile.get("kunya")),
        "nisba": _clean(profile.get("nasab")),
        "laqab": _clean(profile.get("laqab")),
        "birth_year_ah": None,  # not present in the source
        "death_year_ah": _extract_death_year(profile.get("death")),
        "birth_location": _clean(profile.get("city")),  # primary city of activity
        "death_location": None,
        "generation": _generation(profile, grade),
        "gender": None,  # not recorded by Itqan
        "trustworthiness": trustworthiness,
        "bio_text": _clean(profile.get("grade_ar")),  # the jarh-wa-ta'dil verdict text
        "external_id": profile_id,
    }


def _parse_profiles_file(path: Path) -> list[dict[str, object | None]]:
    """Parse one ``profiles_<grade>.json`` bucket into bio rows."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object | None]] = []
    for raw_id, profile in data.items():
        if not isinstance(profile, dict):
            continue
        row = _parse_profile(raw_id, profile)
        if row is not None:
            rows.append(row)
    logger.info("itqan_bucket_parsed", file=path.name, rows=len(rows))
    return rows


def run(raw_dir: Path, staging_dir: Path) -> Path:
    """Parse Itqan rijal buckets into ``narrators_bio_itqan.parquet``."""
    source_dir = raw_dir / "itqan"
    if not source_dir.exists():
        msg = f"Source directory not found: {source_dir}"
        raise FileNotFoundError(msg)

    bucket_files = sorted(source_dir.glob("profiles_*.json"))
    if not bucket_files:
        msg = f"No profiles_*.json buckets under {source_dir}"
        raise FileNotFoundError(msg)

    rows: list[dict[str, object | None]] = []
    seen_bio_ids: set[str] = set()
    for bucket in bucket_files:
        for row in _parse_profiles_file(bucket):
            bio_id = str(row["bio_id"])
            if bio_id in seen_bio_ids:
                logger.warning("itqan_duplicate_bio_id", bio_id=bio_id)
                continue
            seen_bio_ids.add(bio_id)
            rows.append(row)

    if not rows:
        msg = "No valid Itqan narrator bios parsed"
        raise ValueError(msg)

    arrays = {field.name: [r[field.name] for r in rows] for field in NARRATOR_BIO_SCHEMA}
    table = pa.table(arrays, schema=NARRATOR_BIO_SCHEMA)

    out_path = staging_dir / "narrators_bio_itqan.parquet"
    write_parquet(table, out_path, schema=NARRATOR_BIO_SCHEMA)
    logger.info("itqan_bios_parsed", total=len(rows), buckets=len(bucket_files))
    return out_path
