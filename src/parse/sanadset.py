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
dedup (B2 / da#220) and narrator re-segmentation (B3 / da#221) are deliberately
out of scope here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa

from src.config import get_settings
from src.parse.base import (
    generate_source_id,
    read_csv_robust,
    safe_int,
    safe_str,
    write_parquet,
)
from src.parse.identity import ID_DELIMITER
from src.parse.schemas import (
    COLLECTION_SCHEMA,
    HADITH_SCHEMA,
    NARRATOR_BIO_SCHEMA,
    NARRATOR_MENTION_SCHEMA,
)
from src.utils.arabic import extract_transmission_phrases, normalize_arabic
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
    """Extract narrator mentions from SANAD text containing NAR tags.

    Returns list of dicts matching NARRATOR_MENTION_SCHEMA fields.
    """
    mentions: list[dict[str, str | int | None]] = []
    nar_matches = list(_NAR_RE.finditer(sanad_text))

    for idx, match in enumerate(nar_matches):
        name_raw = match.group(1).strip()
        if not name_raw:
            continue

        name_ar_normalized = normalize_arabic(name_raw)
        mention_id = generate_source_id(_SOURCE_CORPUS, "mention", source_hadith_id, str(idx))

        # Extract transmission method from text between previous NAR end and current NAR start
        transmission_method: str | None = None
        if idx > 0:
            prev_end = nar_matches[idx - 1].end()
            between_text = sanad_text[prev_end : match.start()]
        else:
            between_text = sanad_text[: match.start()]

        if between_text.strip():
            phrases = extract_transmission_phrases(between_text)
            if phrases:
                transmission_method = phrases[0][2]  # label of first match

        mentions.append(
            {
                "mention_id": mention_id,
                "source_hadith_id": source_hadith_id,
                "source_corpus": _SOURCE_CORPUS,
                "position_in_chain": idx,
                "name_ar": name_raw,
                "name_en": None,
                "name_ar_normalized": name_ar_normalized,
                "transmission_method": transmission_method,
            }
        )

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
) -> tuple[list[dict[str, object]], list[dict[str, str | int | None]], int]:
    """Process a chunk of rows, returning hadith records, narrator mentions, and malformed count."""
    hadiths: list[dict[str, object]] = []
    mentions: list[dict[str, str | int | None]] = []
    malformed_count = 0

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
            try:
                row_mentions = _extract_narrator_mentions(isnad_raw_ar, source_id)
                mentions.extend(row_mentions)
            except Exception:  # noqa: BLE001
                malformed_count += 1
                logger.debug("malformed_nar_tags", source_id=source_id)

    return hadiths, mentions, malformed_count


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
            hadiths, mentions, malformed = _process_chunk(
                chunk,
                collection_name,
                row_offset=global_row_offset + chunk_start,
                has_books=has_books,
            )
            all_hadiths.extend(hadiths)
            all_mentions.extend(mentions)
            total_malformed += malformed

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

    logger.info(
        "sanadset_parse_quality",
        total_rows=total_rows,
        parsed_hadiths=len(all_hadiths),
        valid_sanad_pct=round(valid_sanad_pct, 1),
        narrator_mentions=len(all_mentions),
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
