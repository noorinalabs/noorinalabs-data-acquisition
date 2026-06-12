"""Parse the Itqan rijal dataset into staging Parquet.

Produces three output files from the per-grade profile buckets downloaded by
``src/acquire/itqan.py``:

* ``narrators_bio_itqan.parquet``    (NARRATOR_BIO_SCHEMA)   — the bios (da#92a).
* ``network_edges_itqan.parquet``    (NETWORK_EDGE_SCHEMA)   — teacher↔student
  transmission edges built from each profile's ``teachers``/``students`` id
  lists (da#93). Loaded as ``STUDIED_UNDER`` edges, mirroring muhaddithat.
* ``narrator_aliases_itqan.parquet`` (NARRATOR_ALIAS_SCHEMA) — name variants
  from each profile's ``namings`` array (da#94), keyed to the same canonical
  identity the bio mints so the resolve stage attaches them as aliases.

Each bucket file (``profiles_<grade>.json``) is a JSON object keyed by the
narrator id; every value is a profile dict. The jarh-wa-ta'dil grade is carried
both by the filename bucket and the per-profile ``grade_en`` field (they agree);
we trust the per-profile field and fall back to the bucket.

Edges (da#93): Itqan is a rijal database — it has no hadith-scoped isnad chains,
so there is nothing to emit as NARRATOR_MENTION rows (those need a
``source_hadith_id`` + ``position_in_chain``). Instead each profile carries the
ids of its ``teachers`` and ``students``; the transmission relation is
``student STUDIED_UNDER teacher`` (``from_narrator`` = student, ``to_narrator`` =
teacher), exactly the orientation the muhaddithat NETWORK_EDGE loader uses. A
profile P thus contributes ``(P -> t)`` for each teacher t and ``(s -> P)`` for
each student s; the two perspectives are deduplicated on the id pair.

Aliases (da#94): each profile's ``namings`` array lists alternate spellings of
the narrator's name. We emit one alias row per distinct variant, tagged with the
profile's canonical normalized name so ``bio_promote`` can union the variants
onto the matching ``nar:`` record without re-reading the source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa

from src.parse.base import safe_str, write_parquet
from src.parse.schemas import (
    NARRATOR_ALIAS_SCHEMA,
    NARRATOR_BIO_SCHEMA,
    NETWORK_EDGE_SCHEMA,
)
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


def _id_list(value: object) -> list[str]:
    """Coerce a ``teachers``/``students`` field (ints or strs) to clean id strings."""
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        s = safe_str(item)
        if s:
            ids.append(s)
    return ids


def _iter_profiles(bucket_files: list[Path]) -> list[tuple[str, dict[str, object]]]:
    """Load every bucket into a flat ``(profile_id, profile)`` list.

    ``id`` is globally unique across buckets (verified — da#92a); the first
    occurrence of a profile id wins so re-running over an overlapping bucket set
    does not double-count. A profile with no id is dropped (it can key neither a
    bio nor a transmission edge).
    """
    profiles: list[tuple[str, dict[str, object]]] = []
    seen: set[str] = set()
    for bucket in bucket_files:
        data = json.loads(bucket.read_text(encoding="utf-8"))
        count = 0
        for raw_id, profile in data.items():
            if not isinstance(profile, dict):
                continue
            pid = safe_str(profile.get("id")) or safe_str(raw_id)
            if pid is None:
                continue
            if pid in seen:
                logger.warning("itqan_duplicate_profile_id", profile_id=pid)
                continue
            seen.add(pid)
            profiles.append((pid, profile))
            count += 1
        logger.info("itqan_bucket_parsed", file=bucket.name, profiles=count)
    return profiles


def _build_edges(
    profiles: list[tuple[str, dict[str, object]]],
    id_to_name: dict[str, str],
) -> list[dict[str, str | None]]:
    """Build deduplicated ``student -> teacher`` STUDIED_UNDER edges (da#93).

    For each profile P: ``P -> t`` for every teacher ``t`` in ``P.teachers``, and
    ``s -> P`` for every student ``s`` in ``P.students``. ``from`` is always the
    student, ``to`` the teacher (the orientation the NETWORK_EDGE loader maps to
    ``(student)-[:STUDIED_UNDER]->(teacher)``). The same edge surfaces from both
    endpoints, so it is deduplicated on the ``(from_id, to_id)`` id pair. The name
    columns fall back to the bare id when a referenced profile is absent from the
    parsed slice, keeping them non-null per NETWORK_EDGE_SCHEMA.
    """
    edges: list[dict[str, str | None]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _add(student_id: str, teacher_id: str) -> None:
        if student_id == teacher_id:
            return
        key = (student_id, teacher_id)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        edges.append(
            {
                "from_narrator_name": id_to_name.get(student_id, student_id),
                "to_narrator_name": id_to_name.get(teacher_id, teacher_id),
                "hadith_id": None,  # rijal transmission network — not hadith-scoped
                "source": SOURCE,
                "from_external_id": student_id,
                "to_external_id": teacher_id,
            }
        )

    for pid, profile in profiles:
        for teacher_id in _id_list(profile.get("teachers")):
            _add(pid, teacher_id)
        for student_id in _id_list(profile.get("students")):
            _add(student_id, pid)

    return edges


def _build_aliases(profiles: list[tuple[str, dict[str, object]]]) -> list[dict[str, str]]:
    """Build name-variant alias rows from each profile's ``namings`` array (da#94).

    One row per distinct variant whose normalized form differs from the
    narrator's primary normalized name — the primary spelling is not an alias of
    itself, and exact duplicate variants are collapsed. ``canonical_name_ar_normalized``
    carries the primary normalized name so ``bio_promote`` can key each variant
    onto the matching ``nar:`` record.
    """
    rows: list[dict[str, str]] = []
    for pid, profile in profiles:
        name_ar = _clean(profile.get("full_name"))
        if name_ar is None:
            continue  # no canonical name to attach the variants to
        canonical_norm = normalize_arabic(name_ar)
        namings = profile.get("namings")
        if not isinstance(namings, list):
            continue
        bio_id = f"{SOURCE}:{pid}"
        seen: set[str] = set()
        for raw_variant in namings:
            variant = _clean(raw_variant)
            if variant is None:
                continue
            variant_norm = normalize_arabic(variant)
            if not variant_norm or variant_norm == canonical_norm or variant_norm in seen:
                continue
            seen.add(variant_norm)
            rows.append(
                {
                    "bio_id": bio_id,
                    "source": SOURCE,
                    "canonical_name_ar_normalized": canonical_norm,
                    "alias": variant,
                    "alias_normalized": variant_norm,
                }
            )
    return rows


def run(raw_dir: Path, staging_dir: Path) -> list[Path]:
    """Parse Itqan rijal buckets into bio, network-edge, and alias Parquet.

    Returns ``[bios, edges, aliases]``. ``src.parse.run_all`` flattens a list
    result, so a single adapter emitting three staging files is a first-class
    return shape (``ParseOutput``).
    """
    source_dir = raw_dir / "itqan"
    if not source_dir.exists():
        msg = f"Source directory not found: {source_dir}"
        raise FileNotFoundError(msg)

    bucket_files = sorted(source_dir.glob("profiles_*.json"))
    if not bucket_files:
        msg = f"No profiles_*.json buckets under {source_dir}"
        raise FileNotFoundError(msg)

    profiles = _iter_profiles(bucket_files)

    # Bios (da#92a) — every named profile becomes a NARRATOR_BIO row.
    bio_rows: list[dict[str, object | None]] = []
    for pid, profile in profiles:
        row = _parse_profile(pid, profile)
        if row is not None:
            bio_rows.append(row)
    if not bio_rows:
        msg = "No valid Itqan narrator bios parsed"
        raise ValueError(msg)
    bio_arrays = {f.name: [r[f.name] for r in bio_rows] for f in NARRATOR_BIO_SCHEMA}
    bio_table = pa.table(bio_arrays, schema=NARRATOR_BIO_SCHEMA)
    bio_path = staging_dir / "narrators_bio_itqan.parquet"
    write_parquet(bio_table, bio_path, schema=NARRATOR_BIO_SCHEMA)

    # Network edges (da#93) — teacher/student id lists -> STUDIED_UNDER edges.
    id_to_name: dict[str, str] = {}
    for pid, profile in profiles:
        name = _clean(profile.get("full_name"))
        if name is not None:
            id_to_name[pid] = name
    edges = _build_edges(profiles, id_to_name)
    edge_arrays = {f.name: [e[f.name] for e in edges] for f in NETWORK_EDGE_SCHEMA}
    edge_table = pa.table(edge_arrays, schema=NETWORK_EDGE_SCHEMA)
    edge_path = staging_dir / "network_edges_itqan.parquet"
    write_parquet(edge_table, edge_path, schema=NETWORK_EDGE_SCHEMA)

    # Name variants (da#94) -> identity aliases consumed by bio_promote.
    aliases = _build_aliases(profiles)
    alias_arrays = {f.name: [a[f.name] for a in aliases] for f in NARRATOR_ALIAS_SCHEMA}
    alias_table = pa.table(alias_arrays, schema=NARRATOR_ALIAS_SCHEMA)
    alias_path = staging_dir / "narrator_aliases_itqan.parquet"
    write_parquet(alias_table, alias_path, schema=NARRATOR_ALIAS_SCHEMA)

    logger.info(
        "itqan_parsed",
        buckets=len(bucket_files),
        profiles=len(profiles),
        bios=len(bio_rows),
        edges=len(edges),
        aliases=len(aliases),
    )
    return [bio_path, edge_path, alias_path]
