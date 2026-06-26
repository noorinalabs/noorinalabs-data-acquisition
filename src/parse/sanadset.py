"""Parse the Sanadset Kaggle dataset into staging Parquet files.

The hadith dataset uses XML-style inline tags:
- ``<SANAD>...</SANAD>`` — isnad (chain of narration)
- ``<MATN>...</MATN>`` — matn (hadith text body)
- ``<NAR>...</NAR>`` — narrator name within the SANAD

Produces four Parquet files:
- ``hadiths_sanadset.parquet`` — hadith records
- ``narrator_mentions_sanadset.parquet`` — narrator mentions per chain
- ``narrators_bio_kaggle.parquet`` — narrator biographies from narrators dataset
- ``collections_sanadset.parquet`` — one Collection per book (da#219 / Path B-B1)

Collection emission (da#219, Path B / B1)
-----------------------------------------
Before B1 ``parse_sanadset`` emitted no ``collections_*`` file, so
``load_nodes._load_collections`` created zero sanadset ``Collection`` nodes and
``load_edges._APPEARS_IN_QUERY`` — a *non-creating* ``MATCH (c:Collection)`` —
silently dropped every sanadset ``APPEARS_IN`` edge (the headline orphan defect,
ADR-003). B1 reads the already-acquired ``books.csv`` and emits one ``Collection``
per book so each hadith links to its book's collection. The APPEARS_IN endpoint is
keyed ``col:<corpus>:<collection_name>`` (``load_edges._load_appears_in``), so the
join key is the hadith's ``collection_name``: when ``books.csv`` is present it is
the ``book_id`` (per-book breadth — the reason Path B was chosen over a purge);
otherwise it falls back to the per-CSV-stem name (pre-B1 behavior). Cross-edition
dedup (B2 / da#220) is deliberately out of scope here.

Narrator re-segmentation (da#221, Path B / B3)
----------------------------------------------
The raw ``<NAR>`` tags are a *coarse* per-fragment firehose: a tag may hold a
genuine narrator name, but it may equally hold an honorific/eulogy phrase
(``رضي الله عنه``), a bare transmission verb (``قال`` / ``عن``), a non-Arabic or
vowel-stripped transliteration fragment (``He said`` / ``that``), or a whole
mis-tagged *sub-chain* carrying internal transmission markers. Stored at face
value (pre-B3 behaviour) these polluted the resolved ``Narrator`` table, made
full-text search surface matns as narrator hits (ig#1110), and starved
``STUDIED_UNDER`` (ADR-003). B3 routes every ``<NAR>`` tag through
:func:`_segment_nar_content`, which (1) re-segments a marker-bearing blob into its
constituent narrators via the shared
:func:`src.parse.narrator_extraction.extract_narrator_mentions` segmenter
(dropping an un-segmentable chain blob rather than minting it, da#158), and
(2) filters non-narrator pollution via :func:`_is_narrator_like` — Arabic script
required (drops English / Latin transliteration fragments), no residual
transmission marker, not a whole honorific phrase, and not a bare nasab particle.
Genuine narrators flow through unchanged (the common single-name tag keeps its raw
voweled ``name_ar``). The net effect is a substantial drop in the mention
firehose — the ``narrator_mentions_sanadset`` baseline in
:mod:`src.parse.validate` is recalibrated accordingly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa

from src.config import get_settings
from src.models.enums import DatePrecision
from src.parse.base import (
    generate_source_id,
    read_csv_robust,
    safe_int,
    safe_str,
    write_parquet,
)
from src.parse.collection_metadata import apply_collection_metadata
from src.parse.identity import ID_DELIMITER
from src.parse.narrator_extraction import (
    IsnadSegmentationError,
    extract_narrator_mentions,
)
from src.parse.schemas import (
    COLLECTION_SCHEMA,
    HADITH_SCHEMA,
    NARRATOR_BIO_SCHEMA,
    NARRATOR_MENTION_SCHEMA,
)
from src.utils.arabic import (
    contains_transmission_marker,
    extract_transmission_phrases,
    is_arabic,
    normalize_arabic,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["parse_sanadset"]

# ---------------------------------------------------------------------------
# Compiled regexes for XML-style tag extraction
# ---------------------------------------------------------------------------

_SANAD_RE: re.Pattern[str] = re.compile(r"<SANAD>(.*?)</SANAD>", re.DOTALL)
_MATN_RE: re.Pattern[str] = re.compile(r"<MATN>(.*?)</MATN>", re.DOTALL)
_NAR_RE: re.Pattern[str] = re.compile(r"<NAR>(.*?)</NAR>", re.DOTALL)

_CHUNK_SIZE = 50_000
_SOURCE_CORPUS = "sanadset"
_SECT = "sunni"
_BIO_SOURCE = "kaggle_narrators"
# books.csv (Mendeley 5xth87zwb5) sits alongside sanadset.csv in the raw dir and
# maps each hadith's ``book_id`` to its collection. It is NOT a hadith CSV, so it
# is excluded from the hadith glob and consumed as the book→collection mapping.
_BOOKS_FILENAME = "books.csv"

# ---------------------------------------------------------------------------
# Narrator re-segmentation / pollution filtering (da#221, Path B / B3)
# ---------------------------------------------------------------------------

# Minimum substantive length (Arabic letters, spaces excluded) for a span to be a
# plausible narrator name. The shortest real classical names — علي / عمر / زيد —
# normalize to 3 letters; anything shorter is a particle or stray glyph, not a
# name.
_MIN_NARRATOR_CHARS = 3

# Whole-tag honorific / eulogy phrases that routinely ride inside ``<NAR>`` tags
# but name no narrator (``رضي الله عنه`` "may God be pleased with him", the
# ṣalawāt, …). Stored normalized so a voweled tag matches after
# :func:`normalize_arabic`. Only an *entire* tag equal to one of these is dropped;
# an honorific trailing a real name leaves the name intact.
_RAW_NON_NARRATOR_PHRASES: frozenset[str] = frozenset(
    {
        "رضي الله عنه",
        "رضي الله عنها",
        "رضي الله عنهم",
        "رضي الله عنهما",
        "صلى الله عليه وسلم",
        "عليه السلام",
        "عليها السلام",
        "عليهم السلام",
        "رحمه الله",
        "رحمها الله",
        "عز وجل",
        "سبحانه وتعالى",
        "تعالى",
    }
)
_NON_NARRATOR_PHRASES: frozenset[str] = frozenset(
    normalize_arabic(p) for p in _RAW_NON_NARRATOR_PHRASES
)

# Bare nasab / kunya structural particles. A span made up of nothing but these
# (``بن`` / ``ابن`` / ``ابو`` …) is a dangling connective, not a narrator. Stored
# normalized for diacritic/orthographic-insensitive comparison.
_RAW_STRUCTURAL_PARTICLES: frozenset[str] = frozenset(
    {"بن", "ابن", "أبو", "أبا", "أبي", "أم", "عبد", "ال", "بنت"}
)
_STRUCTURAL_PARTICLES: frozenset[str] = frozenset(
    normalize_arabic(p) for p in _RAW_STRUCTURAL_PARTICLES
)


def _is_narrator_like(name: str) -> bool:
    """Return True if *name* is a plausible narrator entity (da#221, B3).

    Rejects the pollution classes the coarse ``<NAR>`` firehose mints as
    ``Narrator`` nodes (ADR-003):

    * non-Arabic spans — English fragments (``He said``/``that``) and
      vowel-stripped Latin transliterations;
    * spans still carrying a transmission marker — a residual / partial-chain
      fragment, never a clean name (da#158);
    * whole honorific/eulogy phrases that name no narrator;
    * bare nasab particles with no actual name token;
    * sub-substantive spans below :data:`_MIN_NARRATOR_CHARS`.
    """
    if not name or not name.strip():
        return False
    text = name.strip()
    # Arabic script required — drops English fragments and Latin transliterations.
    if not is_arabic(text):
        return False
    # A residual transmission marker means a fragment / partial blob, not a name.
    if contains_transmission_marker(text):
        return False
    normalized = normalize_arabic(text)
    if normalized in _NON_NARRATOR_PHRASES:
        return False
    tokens = normalized.split()
    if tokens and all(tok in _STRUCTURAL_PARTICLES for tok in tokens):
        return False
    if len(normalized.replace(" ", "")) < _MIN_NARRATOR_CHARS:
        return False
    return True


def _segment_nar_content(content: str) -> list[tuple[str, str, str | None]]:
    """Re-segment one ``<NAR>`` tag into clean narrator mentions (da#221, B3).

    Returns ``(name_ar, name_ar_normalized, transmission_method)`` tuples — zero
    when the tag is entirely pollution, one for the common clean single-name tag,
    or several when a single tag mis-holds a whole sub-chain.

    * A tag whose own text carries a transmission marker is a mis-tagged blob: it
      is re-segmented via the shared
      :func:`src.parse.narrator_extraction.extract_narrator_mentions` Arabic
      segmenter. An *un-segmentable* marker-bearing blob is dropped (never minted
      as a whole-chain "narrator", da#158); the split pieces are kept in their
      normalized form (a blob has no single clean voweled form anyway).
    * A clean single-name tag keeps its raw, voweled ``name_ar`` and is accepted
      only if :func:`_is_narrator_like`.
    """
    stripped = content.strip()
    if not stripped:
        return []
    if contains_transmission_marker(stripped):
        try:
            spans = extract_narrator_mentions(stripped, "ar")
        except IsnadSegmentationError:
            # Un-segmentable chain blob → drop rather than pollute the table.
            return []
        return [
            (span.name, span.name, span.transmission_method)
            for span in spans
            if _is_narrator_like(span.name)
        ]
    normalized = normalize_arabic(stripped)
    if not _is_narrator_like(normalized):
        return []
    return [(stripped, normalized, None)]


def _extract_tag(pattern: re.Pattern[str], text: str) -> str | None:
    """Extract first match of a tag pattern, returning stripped content or None."""
    m = pattern.search(text)
    if m:
        content = m.group(1).strip()
        return content if content else None
    return None


def _extract_narrator_mentions(
    sanad_text: str,
    source_hadith_id: str,
) -> list[dict[str, str | int | None]]:
    """Extract clean narrator mentions from SANAD text containing NAR tags.

    Each ``<NAR>`` tag is re-segmented and pollution-filtered by
    :func:`_segment_nar_content` (da#221, B3): genuine narrators are emitted, a
    mis-tagged sub-chain is split into its constituents, and non-narrator
    fragments (honorifics, bare transmission verbs, English / transliteration
    fragments, dangling particles) are dropped. ``position_in_chain`` is numbered
    over the *emitted* mentions so it stays a gap-free chain order even when tags
    are dropped or split. Returns list of dicts matching NARRATOR_MENTION_SCHEMA.
    """
    mentions: list[dict[str, str | int | None]] = []
    nar_matches = list(_NAR_RE.finditer(sanad_text))

    position = 0
    for idx, match in enumerate(nar_matches):
        # Transmission method carried by the text between the previous NAR's end
        # and this tag's start (e.g. the ``عن`` joining two narrators).
        between_method: str | None = None
        between_text = (
            sanad_text[nar_matches[idx - 1].end() : match.start()]
            if idx > 0
            else sanad_text[: match.start()]
        )
        if between_text.strip():
            phrases = extract_transmission_phrases(between_text)
            if phrases:
                between_method = phrases[0][2]  # label of first match

        for name_ar, name_ar_normalized, span_method in _segment_nar_content(match.group(1)):
            mention_id = generate_source_id(
                _SOURCE_CORPUS, "mention", source_hadith_id, str(position)
            )
            # An internal blob-split carries its own method; otherwise the
            # between-tag transmission phrase applies to this narrator.
            method = span_method if span_method is not None else between_method
            mentions.append(
                {
                    "mention_id": mention_id,
                    "source_hadith_id": source_hadith_id,
                    "source_corpus": _SOURCE_CORPUS,
                    "position_in_chain": position,
                    "name_ar": name_ar,
                    "name_en": None,
                    "name_ar_normalized": name_ar_normalized,
                    "transmission_method": method,
                }
            )
            position += 1

    return mentions


def _parse_books(raw_dir: Path) -> dict[str, dict[str, object]]:
    """Read ``books.csv`` into a ``{book_id: collection-metadata}`` mapping (da#219).

    ``books.csv`` (Mendeley 5xth87zwb5) is the book→collection table acquired
    alongside ``sanadset.csv``. Columns are matched case-insensitively and
    flexibly because the upstream header spelling is not contractually fixed; an
    absent / id-less / unreadable file yields an empty mapping, in which case the
    parser falls back to the per-CSV-stem collection name (pre-B1 behavior). The
    keys are the stringified integer ``book_id`` so they JOIN to the hadith's
    ``collection_name`` (also the ``book_id``) for APPEARS_IN.
    """
    books_path = raw_dir / _BOOKS_FILENAME
    if not books_path.exists():
        return {}
    table, _enc = read_csv_robust(books_path)
    cols: dict[str, str] = {str(c).lower(): str(c) for c in table.column_names}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    id_col = pick("book_id", "id", "bookid")
    if id_col is None:
        logger.warning("books_csv_no_id_column", available=sorted(cols))
        return {}
    name_ar_col = pick("name_ar", "book_name", "name", "title", "arabic_name", "book")
    name_en_col = pick("name_en", "english_name", "name_english", "title_en")
    compiler_col = pick("author", "compiler", "compiler_name", "writer", "musannif")
    count_col = pick(
        "number_of_hadith", "hadith_count", "num_hadith", "count", "total_hadiths", "hadiths"
    )

    books: dict[str, dict[str, object]] = {}
    for i in range(table.num_rows):
        row = {c: table.column(c)[i].as_py() for c in table.column_names}
        book_id = safe_int(row.get(id_col))
        if book_id is None:
            continue
        key = str(book_id)
        name_ar = safe_str(row.get(name_ar_col)) if name_ar_col else None
        name_en = safe_str(row.get(name_en_col)) if name_en_col else None
        books[key] = {
            "name_ar": name_ar,
            # COLLECTION_SCHEMA.name_en is non-nullable: fall back to the Arabic
            # title, then to a deterministic per-book label, so it is never empty.
            "name_en": name_en or name_ar or f"Sanadset book {key}",
            "compiler_name": safe_str(row.get(compiler_col)) if compiler_col else None,
            "total_hadiths": safe_int(row.get(count_col)) if count_col else None,
        }
    logger.info("books_csv_parsed", books=len(books))
    return books


def _resolve_collection_name(book_num: int | None, default_name: str, has_books: bool) -> str:
    """Collection name (APPEARS_IN join key) for a hadith row (da#219).

    With ``books.csv`` present the per-book ``book_id`` is the collection (so the
    corpus keeps its collection breadth — the reason Path B was chosen over a
    purge); without it, the pre-B1 per-CSV-stem ``default_name`` is kept.
    """
    if has_books and book_num is not None:
        return str(book_num)
    return default_name


def _collection_row(
    collection_name: str, total_hadiths: int, books: dict[str, dict[str, object]], has_books: bool
) -> dict[str, object]:
    """Build one ``COLLECTION_SCHEMA`` row for a distinct ``collection_name``.

    ``collection_id`` is ``generate_source_id(corpus, collection_name)`` —
    identical to the key ``load_edges._load_appears_in`` rebuilds from the hadith
    (``f"{corpus}:{collection_name}"``), so ``collection_node_id`` agrees on both
    endpoints and the APPEARS_IN MATCH resolves.
    """
    meta = books.get(collection_name)
    if meta is not None:
        name_ar = meta["name_ar"]
        name_en = meta["name_en"]
        compiler_name = meta["compiler_name"]
    else:
        name_ar = None
        # A book_id with no books.csv row (has_books) gets a deterministic label;
        # the per-CSV-stem fallback keeps the stem itself as a human name.
        name_en = f"Sanadset book {collection_name}" if has_books else collection_name
        compiler_name = None
    return {
        "collection_id": generate_source_id(_SOURCE_CORPUS, collection_name),
        "name_ar": name_ar,
        "name_en": name_en,
        "compiler_name": compiler_name,
        "compilation_year_ah": None,
        "sect": _SECT,
        "total_hadiths": total_hadiths,
        "source_corpus": _SOURCE_CORPUS,
    }


def _process_chunk(
    rows: list[dict[str, object]],
    collection_name: str,
    row_offset: int = 0,
    *,
    has_books: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, str | int | None]], int, int]:
    """Process a chunk of rows.

    Returns ``(hadiths, mentions, malformed_count, raw_nar_count)``. ``raw_nar_count``
    is the number of raw ``<NAR>`` tags seen before B3 re-segmentation / filtering;
    comparing it to ``len(mentions)`` quantifies how much coarse-tag pollution the
    B3 filter removed (da#221).
    """
    hadiths: list[dict[str, object]] = []
    mentions: list[dict[str, str | int | None]] = []
    malformed_count = 0
    raw_nar_count = 0

    for idx, row in enumerate(rows):
        full_text = safe_str(row.get("hadith") or row.get("text") or row.get("Hadith"))
        if full_text is None:
            continue

        hadith_num = safe_int(row.get("hadith_id") or row.get("id") or row.get("Hadith_ID"))
        book_num = safe_int(row.get("book_id") or row.get("book") or row.get("Book_ID"))
        row_collection = _resolve_collection_name(book_num, collection_name, has_books)

        source_id = generate_source_id(
            _SOURCE_CORPUS,
            row_collection,
            str(book_num or 0),
            str(hadith_num or 0),
            str(row_offset + idx),
        )

        # Extract SANAD and MATN content
        sanad_text = _extract_tag(_SANAD_RE, full_text)
        matn_text = _extract_tag(_MATN_RE, full_text)

        # Handle "No SANAD" rows
        isnad_raw_ar: str | None = None
        if sanad_text and sanad_text.lower() != "no sanad":
            isnad_raw_ar = sanad_text

        hadiths.append(
            {
                "source_id": source_id,
                "source_corpus": _SOURCE_CORPUS,
                "collection_name": row_collection,
                "book_number": book_num,
                "chapter_number": None,
                "hadith_number": hadith_num,
                "matn_ar": matn_text,
                "matn_en": None,
                "isnad_raw_ar": isnad_raw_ar,
                "isnad_raw_en": None,
                "full_text_ar": full_text,
                "full_text_en": None,
                "grade": safe_str(row.get("grade") or row.get("Grade")),
                "chapter_name_ar": safe_str(row.get("chapter") or row.get("Chapter")),
                "chapter_name_en": None,
                "sect": "sunni",
            }
        )

        # Extract narrator mentions from SANAD if available
        if isnad_raw_ar:
            raw_nar_count += len(_NAR_RE.findall(isnad_raw_ar))
            try:
                row_mentions = _extract_narrator_mentions(isnad_raw_ar, source_id)
                mentions.extend(row_mentions)
            except Exception:  # noqa: BLE001
                malformed_count += 1
                logger.debug("malformed_nar_tags", source_id=source_id)

    return hadiths, mentions, malformed_count, raw_nar_count


def _parse_narrators_bio(narrators_dir: Path) -> pa.Table | None:
    """Parse narrator biography CSV files dynamically based on available headers."""
    csv_files = list(narrators_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("no_narrator_bio_csv", dir=str(narrators_dir))
        return None

    all_bios: list[dict[str, object]] = []

    for csv_file in csv_files:
        table, _enc = read_csv_robust(csv_file)
        column_names = [c.lower() for c in table.column_names]

        # Build a mapping from expected fields to actual column names
        field_map: dict[str, str] = {}
        for actual_name in table.column_names:
            lower = actual_name.lower()
            if lower in ("name", "narrator_name", "name_ar", "narrator"):
                field_map["name_ar"] = actual_name
            elif lower in ("english_name", "name_en", "name_english"):
                field_map["name_en"] = actual_name
            elif lower in ("kunya", "kunyah"):
                field_map["kunya"] = actual_name
            elif lower in ("nisba", "nisbah"):
                field_map["nisba"] = actual_name
            elif lower in ("laqab",):
                field_map["laqab"] = actual_name
            elif lower in ("birth_year", "birth", "birth_year_ah"):
                field_map["birth_year_ah"] = actual_name
            elif lower in ("death_year", "death", "death_year_ah"):
                field_map["death_year_ah"] = actual_name
            elif lower in ("birth_location", "birth_place"):
                field_map["birth_location"] = actual_name
            elif lower in ("death_location", "death_place"):
                field_map["death_location"] = actual_name
            elif lower in ("generation", "tabaqa", "tabaqat"):
                field_map["generation"] = actual_name
            elif lower in ("gender", "sex"):
                field_map["gender"] = actual_name
            elif lower in (
                "trustworthiness",
                "rank",
                "grade",
                "jarh_tadil",
                "reliability",
            ):
                field_map["trustworthiness"] = actual_name
            elif lower in ("bio", "biography", "bio_text", "description"):
                field_map["bio_text"] = actual_name
            elif lower in ("id", "narrator_id", "external_id"):
                field_map["external_id"] = actual_name

        logger.info(
            "narrator_bio_columns",
            file=csv_file.name,
            available=column_names,
            mapped=list(field_map.keys()),
        )

        for i in range(table.num_rows):
            row_dict: dict[str, object] = {
                col: table.column(col)[i].as_py() for col in table.column_names
            }

            name_ar = safe_str(row_dict.get(field_map.get("name_ar", "")))
            ext_id = safe_str(row_dict.get(field_map.get("external_id", "")))
            # The bio_id is a narrator-biography provenance key namespaced by the
            # *bio source* (``kaggle_narrators``), which is NOT a hadith
            # ``SourceCorpus``. Building it with ``generate_source_id`` would trip
            # that helper's da#82 fail-fast on an unknown corpus, so we join the
            # key directly with the shared delimiter instead.
            bio_id = ID_DELIMITER.join([_BIO_SOURCE, csv_file.stem, str(ext_id or i)])
            name_ar_norm = normalize_arabic(name_ar) if name_ar else None

            all_bios.append(
                {
                    "bio_id": bio_id,
                    "source": _BIO_SOURCE,
                    "name_ar": name_ar,
                    "name_en": safe_str(row_dict.get(field_map.get("name_en", ""))),
                    "name_ar_normalized": name_ar_norm,
                    "name_en_normalized": None,
                    "kunya": safe_str(row_dict.get(field_map.get("kunya", ""))),
                    "nisba": safe_str(row_dict.get(field_map.get("nisba", ""))),
                    "laqab": safe_str(row_dict.get(field_map.get("laqab", ""))),
                    "birth_year_ah": safe_int(row_dict.get(field_map.get("birth_year_ah", ""))),
                    "death_year_ah": safe_int(row_dict.get(field_map.get("death_year_ah", ""))),
                    # Sanadset carries only a scalar point year (no span/openness
                    # notation), so the da#164 bound columns stay null with a concrete
                    # "unknown" precision (never null). Promoting these point years to
                    # EXACT bounds is reconcile/fallback territory (da#165/#166).
                    "birth_year_ah_earliest": None,
                    "birth_year_ah_latest": None,
                    "birth_date_precision": DatePrecision.UNKNOWN.value,
                    "death_year_ah_earliest": None,
                    "death_year_ah_latest": None,
                    "death_date_precision": DatePrecision.UNKNOWN.value,
                    "birth_location": safe_str(row_dict.get(field_map.get("birth_location", ""))),
                    "death_location": safe_str(row_dict.get(field_map.get("death_location", ""))),
                    "generation": safe_str(row_dict.get(field_map.get("generation", ""))),
                    "gender": safe_str(row_dict.get(field_map.get("gender", ""))),
                    "trustworthiness": safe_str(row_dict.get(field_map.get("trustworthiness", ""))),
                    "bio_text": safe_str(row_dict.get(field_map.get("bio_text", ""))),
                    "external_id": str(ext_id) if ext_id else None,
                }
            )

    if not all_bios:
        return None

    return pa.Table.from_pylist(all_bios, schema=NARRATOR_BIO_SCHEMA)


def parse_sanadset(
    raw_dir: Path | None = None,
    staging_dir: Path | None = None,
) -> dict[str, Path]:
    """Parse Sanadset raw CSV files into staging Parquet.

    Parameters
    ----------
    raw_dir
        Directory containing downloaded Sanadset CSV files.
        Defaults to ``{data_raw_dir}/sanadset/``.
    staging_dir
        Output directory for Parquet files.
        Defaults to ``{data_staging_dir}/``.

    Returns
    -------
    dict[str, Path]
        Mapping of output name to Parquet file path.
    """
    settings = get_settings()
    if raw_dir is None:
        raw_dir = settings.data_raw_dir / "sanadset"
    if staging_dir is None:
        staging_dir = settings.data_staging_dir

    staging_dir.mkdir(parents=True, exist_ok=True)

    # books.csv is the book→collection mapping, not a hadith CSV — consume it
    # separately (da#219) and exclude it from the hadith glob so it is never
    # mis-parsed as hadith rows.
    books = _parse_books(raw_dir)
    has_books = bool(books)

    csv_files = sorted(p for p in raw_dir.glob("*.csv") if p.name.lower() != _BOOKS_FILENAME)
    if not csv_files:
        msg = f"No CSV files found in {raw_dir}"
        raise FileNotFoundError(msg)

    all_hadiths: list[dict[str, object]] = []
    all_mentions: list[dict[str, str | int | None]] = []
    total_malformed = 0
    total_raw_nar = 0
    total_rows = 0
    global_row_offset = 0

    for csv_file in csv_files:
        logger.info("parsing_csv", file=csv_file.name)
        table, _enc = read_csv_robust(csv_file)
        total_rows += table.num_rows

        # Convert to list of dicts and process in chunks
        rows = [
            {col: table.column(col)[i].as_py() for col in table.column_names}
            for i in range(table.num_rows)
        ]

        collection_name = csv_file.stem.lower().replace(" ", "_")

        for chunk_start in range(0, len(rows), _CHUNK_SIZE):
            chunk = rows[chunk_start : chunk_start + _CHUNK_SIZE]
            hadiths, mentions, malformed, raw_nar = _process_chunk(
                chunk,
                collection_name,
                row_offset=global_row_offset + chunk_start,
                has_books=has_books,
            )
            all_hadiths.extend(hadiths)
            all_mentions.extend(mentions)
            total_malformed += malformed
            total_raw_nar += raw_nar

            logger.info(
                "chunk_processed",
                file=csv_file.name,
                offset=chunk_start,
                hadiths=len(hadiths),
                mentions=len(mentions),
            )

        global_row_offset += len(rows)

    # Data quality logging
    valid_sanad_count = sum(1 for h in all_hadiths if h["isnad_raw_ar"] is not None)
    valid_sanad_pct = (valid_sanad_count / len(all_hadiths) * 100) if all_hadiths else 0
    avg_narrators = len(all_mentions) / valid_sanad_count if valid_sanad_count > 0 else 0
    # B3 (da#221) observability: how much coarse-tag pollution the re-segmentation
    # filter removed. total_raw_nar counts raw <NAR> tags; len(all_mentions) is the
    # clean post-filter mention count. A positive filtered_pct is the firehose drop.
    filtered_nar = total_raw_nar - len(all_mentions)
    filtered_nar_pct = (filtered_nar / total_raw_nar * 100) if total_raw_nar else 0

    logger.info(
        "sanadset_parse_quality",
        total_rows=total_rows,
        parsed_hadiths=len(all_hadiths),
        valid_sanad_pct=round(valid_sanad_pct, 1),
        raw_nar_tags=total_raw_nar,
        narrator_mentions=len(all_mentions),
        filtered_nar=filtered_nar,
        filtered_nar_pct=round(filtered_nar_pct, 1),
        avg_narrators_per_chain=round(avg_narrators, 2),
        malformed_tags=total_malformed,
    )

    # Write hadith Parquet
    outputs: dict[str, Path] = {}

    hadith_table = pa.Table.from_pylist(all_hadiths, schema=HADITH_SCHEMA)
    outputs["hadiths"] = write_parquet(
        hadith_table,
        staging_dir / "hadiths_sanadset.parquet",
        schema=HADITH_SCHEMA,
    )

    # Write collections Parquet (da#219 / Path B-B1): one Collection per distinct
    # collection_name actually present on the parsed hadiths, so every hadith has a
    # matching Collection node and its APPEARS_IN edge loads (the headline orphan
    # defect). total_hadiths reflects what was parsed here, not books.csv's count.
    collection_counts: dict[str, int] = {}
    for hadith in all_hadiths:
        cname = str(hadith["collection_name"])
        collection_counts[cname] = collection_counts.get(cname, 0) + 1

    collection_rows = [
        _collection_row(cname, count, books, has_books)
        for cname, count in sorted(collection_counts.items())
    ]
    # Fill sourced name_ar + expected_count where curated (da#230).
    collection_rows = [apply_collection_metadata(r) for r in collection_rows]
    collections_table = pa.Table.from_pylist(collection_rows, schema=COLLECTION_SCHEMA)
    outputs["collections"] = write_parquet(
        collections_table,
        staging_dir / "collections_sanadset.parquet",
        schema=COLLECTION_SCHEMA,
    )
    logger.info(
        "sanadset_collections_emitted",
        collections=len(collection_rows),
        from_books_csv=has_books,
    )

    # Write narrator mentions Parquet
    if all_mentions:
        mentions_table = pa.Table.from_pylist(all_mentions, schema=NARRATOR_MENTION_SCHEMA)
        outputs["narrator_mentions"] = write_parquet(
            mentions_table,
            staging_dir / "narrator_mentions_sanadset.parquet",
            schema=NARRATOR_MENTION_SCHEMA,
        )

    # Parse and write narrator bios
    narrators_dir = raw_dir / "narrators"
    if narrators_dir.exists():
        bio_table = _parse_narrators_bio(narrators_dir)
        if bio_table is not None:
            outputs["narrators_bio"] = write_parquet(
                bio_table,
                staging_dir / "narrators_bio_kaggle.parquet",
                schema=NARRATOR_BIO_SCHEMA,
            )

    logger.info("sanadset_parse_complete", outputs=list(outputs.keys()))
    return outputs
