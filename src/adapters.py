"""Multi-source adapter registry — the single source of truth for ingest sources.

This module is the capstone of epic da#81 (multi-source hadith ingestion, Sunni +
Shia). The per-source acquire/parse modules, the shared base utilities
(:mod:`src.acquire.base` / :mod:`src.parse.base`), the staging schemas
(:mod:`src.parse.schemas`) and the canonical identity contract
(:mod:`src.parse.identity`) were each delivered in their own per-source light-up
PRs. What was missing — and what this module provides — is the *registry* that
binds them into one declared, enforceable contract:

* every ingest source is one :class:`SourceAdapter` row in :data:`SOURCE_REGISTRY`;
* each row declares its ``corpus`` (the :class:`~src.models.enums.SourceCorpus`
  namespace that keeps ``source_id`` collision-safe — see
  :mod:`src.parse.identity`), its ``sect`` (a :class:`~src.models.enums.Sect`, or
  ``None`` for a multi-sect source that tags ``sect`` per record), where its
  acquire/parse code lives, and its reachability + licensing provenance;
* the package orchestrators (:func:`src.acquire.run_all` / :func:`src.parse.run_all`)
  DERIVE their run order from this one list instead of each hand-maintaining a
  parallel ``SOURCES`` / ``PARSERS`` copy that had to be kept in lockstep.

Adding a source (the pattern every per-source PR follows)::

    1. add a distinct value to ``SourceCorpus`` in ``src.models.enums`` — the ONE
       shared registration surface, one non-overlapping value per source;
    2. write ``src/acquire/<slug>.py`` (``run(raw_dir) -> Path | None``) and
       ``src/parse/<slug>.py`` (``run(raw_dir, staging_dir) -> Path | tuple | list``);
    3. add ONE ``SourceAdapter`` row here.

The coverage invariant in ``tests/test_adapters.py`` fails CI if a ``SourceCorpus``
value has no adapter, so step 1 cannot silently drift from step 3.

The registry is pure data: a ``SourceAdapter`` stores *where* its code lives
(``acquire_module`` / ``parse_module`` / fn names) and resolves it lazily on call.
The deferred import is deliberate — this module is imported BY ``src.acquire`` and
``src.parse`` ``__init__``, so importing their submodules at module load would form
a package-init cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from src.models.enums import Sect, SourceCorpus

__all__ = [
    "ParseOutput",
    "SourceAdapter",
    "SOURCE_REGISTRY",
    "adapter_slugs",
    "get_adapter",
    "iter_adapters",
    "adapters_for_corpus",
    "adapters_for_sect",
    "covered_corpora",
]

# A parser's ``run`` may return a single Path, a tuple/list of Paths, or a dict of
# them; ``src.parse.run_all`` flattens this via ``_normalize_output``.
ParseOutput = Path | tuple[Path, ...] | list[Path] | dict[str, Path]


@dataclass(frozen=True)
class SourceAdapter:
    """One ingest source: its corpus namespace, declared sect, code location, and
    reachability + licensing provenance.

    ``slug`` is the adapter key (e.g. ``"sunnah_scraped"``), and is NOT always the
    corpus: ``sunnah`` (the REST API) and ``sunnah_scraped`` (the web scraper)
    deliberately share :attr:`SourceCorpus.SUNNAH` so the same hadith dedups to one
    graph node (see :mod:`src.parse.identity`). The registry is therefore keyed by
    slug and carries ``corpus`` as a field — several adapters MAY map to one corpus.

    ``sect`` is the source's declared sect, or ``None`` for a multi-sect source
    (``fawaz`` spans collections, ``muhaddithat`` + ``itqan`` cover both traditions)
    whose parser tags ``sect`` per record rather than uniformly.

    The acquire/parse entry points are addressed by module + function name and
    resolved lazily in :meth:`acquire` / :meth:`parse`. ``raw_subdir`` handles the
    one source (sanadset) whose downloader/parser are invoked on a per-source
    subdirectory of ``raw_dir`` rather than ``raw_dir`` itself.
    """

    slug: str
    corpus: SourceCorpus
    sect: Sect | None
    acquire_module: str
    parse_module: str
    reachable: bool
    license_note: str
    description: str
    acquire_fn: str = "run"
    parse_fn: str = "run"
    raw_subdir: str | None = None

    def acquire(self, raw_dir: Path) -> Path | None:
        """Download this source's raw data, returning its output path (or ``None``)."""
        target = raw_dir / self.raw_subdir if self.raw_subdir else raw_dir
        fn = getattr(import_module(f"src.acquire.{self.acquire_module}"), self.acquire_fn)
        result: Path | None = fn(target)
        return result

    def parse(self, raw_dir: Path, staging_dir: Path) -> ParseOutput:
        """Parse this source's raw data into staging Parquet under *staging_dir*."""
        source = raw_dir / self.raw_subdir if self.raw_subdir else raw_dir
        fn = getattr(import_module(f"src.parse.{self.parse_module}"), self.parse_fn)
        result: ParseOutput = fn(source, staging_dir)
        return result


# The canonical run order. Keep this list and ``SourceCorpus`` mutually exhaustive
# (the coverage invariant in tests/test_adapters.py enforces it).
SOURCE_REGISTRY: tuple[SourceAdapter, ...] = (
    SourceAdapter(
        slug="lk",
        corpus=SourceCorpus.LK,
        sect=Sect.SUNNI,
        acquire_module="lk_corpus",
        parse_module="lk_corpus",
        reachable=True,
        license_note="LK-Hadith-Corpus (ShathaTm) — open GitHub CSV, 6 books.",
        description="LK-Hadith-Corpus — 6 Sunni books, English + Arabic.",
    ),
    SourceAdapter(
        slug="sanadset",
        corpus=SourceCorpus.SANADSET,
        sect=Sect.SUNNI,
        acquire_module="sanadset",
        parse_module="sanadset",
        acquire_fn="download_sanadset",
        parse_fn="parse_sanadset",
        raw_subdir="sanadset",
        reachable=True,
        license_note="Sanadset 650K (Mendeley 5xth87zwb5) — open dataset.",
        description="Sanadset 650K — 650,986 narrator records / 926 books (Sunni-heavy).",
    ),
    SourceAdapter(
        slug="thaqalayn",
        corpus=SourceCorpus.THAQALAYN,
        sect=Sect.SHIA,
        acquire_module="thaqalayn",
        parse_module="thaqalayn",
        reachable=True,
        license_note="ThaqalaynAPI (MohammedArab1) — open GitHub / REST.",
        description="ThaqalaynAPI — the Four Books and more (Shia).",
    ),
    SourceAdapter(
        slug="fawaz",
        corpus=SourceCorpus.FAWAZ,
        sect=None,
        acquire_module="fawaz",
        parse_module="fawaz",
        reachable=True,
        license_note="fawazahmed0/hadith-api — open, served via jsDelivr.",
        description="fawazahmed0/hadith-api — multi-collection; sect tagged per collection.",
    ),
    SourceAdapter(
        slug="sunnah",
        corpus=SourceCorpus.SUNNAH,
        sect=Sect.SUNNI,
        acquire_module="sunnah_api",
        parse_module="sunnah_api",
        reachable=False,
        license_note=(
            "Sunnah.com API (api.sunnah.com/v1) — requires an API key; 403 keyless "
            "(da#71, never granted)."
        ),
        description="Sunnah.com REST API — Sunni collections (gated on an API key).",
    ),
    SourceAdapter(
        slug="sunnah_scraped",
        corpus=SourceCorpus.SUNNAH,
        sect=Sect.SUNNI,
        acquire_module="sunnah_scraper",
        parse_module="sunnah_scraped",
        reachable=True,
        license_note=(
            "Sunnah.com public web pages — keyless scrape; shares the 'sunnah' corpus "
            "with the API so the same hadith dedups to one node."
        ),
        description="Sunnah.com web scraper — keyless Sunni first-light source.",
    ),
    SourceAdapter(
        slug="open_hadith",
        corpus=SourceCorpus.OPEN_HADITH,
        sect=Sect.SUNNI,
        acquire_module="open_hadith",
        parse_module="open_hadith",
        reachable=True,
        license_note="Open-Hadith-Data (mhashim6) — open GitHub, 9 books incl. the Six Books.",
        description="Open-Hadith-Data — 9 Sunni books.",
    ),
    SourceAdapter(
        slug="muhaddithat",
        corpus=SourceCorpus.MUHADDITHAT,
        sect=None,
        acquire_module="muhaddithat",
        parse_module="muhaddithat",
        reachable=True,
        license_note="muhaddithat/isnad-datasets — open GitHub CSV.",
        description="muhaddithat — female narrators across both traditions; sect per record.",
    ),
    SourceAdapter(
        slug="itqan",
        corpus=SourceCorpus.ITQAN,
        sect=None,
        acquire_module="itqan",
        parse_module="itqan",
        reachable=True,
        license_note=(
            "Itqan (github R3GENESI5/Itqan) — NO upstream license. Owner-approved for "
            "use (da#92a): non-profit, facts re-expressed in our own schema, cleanly "
            "removable via source_corpus='itqan' provenance."
        ),
        description="Itqan rijal DB — 115,735 narrator profiles from 22 classical texts.",
    ),
    SourceAdapter(
        slug="halimbahae",
        corpus=SourceCorpus.HALIMBAHAE,
        sect=Sect.SUNNI,
        acquire_module="halimbahae",
        parse_module="halimbahae",
        reachable=True,
        license_note=(
            "halimbahae/Hadith — open GitHub, 9 Sunni books incl. the Six Books, "
            "Open Database License (ODbL 1.0) + Database Contents License (DbCL 1.0)."
        ),
        description="halimbahae/Hadith — 9 Sunni books with full Arabic diacritics (tashkeel).",
    ),
    SourceAdapter(
        slug="mis",
        corpus=SourceCorpus.MIS,
        sect=Sect.SUNNI,
        acquire_module="mis",
        parse_module="mis",
        reachable=True,
        license_note=(
            "Multi-IsnadSet (MIS) — Mendeley Data 10.17632/gzprcr93zn.2 (Farooqi et al., "
            "Data in Brief 2024). CC BY 4.0; keyless public download over the same "
            "data.mendeley.com mechanism proven for sanadset."
        ),
        description=(
            "Multi-IsnadSet — Sahih Muslim with the MULTIPLE isnad chains per hadith "
            "made explicit (Sunni); one hadith emits N distinct transmission chains."
        ),
    ),
    SourceAdapter(
        slug="bihar",
        corpus=SourceCorpus.BIHAR,
        sect=Sect.SHIA,
        acquire_module="bihar",
        parse_module="bihar",
        reachable=True,
        license_note=(
            "Bihar al-Anwar via hubeali.com 'Read Online' (bilingual AR+EN) — NOT "
            "carried by Thaqalayn (da#95). NO machine-readable upstream license; "
            "owner-approved (da#95, same posture as Itqan da#92a): non-profit, facts "
            "re-expressed in our own schema, cleanly removable via "
            "source_corpus='bihar' provenance. Polite bounded scrape (robots.txt OK)."
        ),
        description="Bihar al-Anwar — al-Majlisi's ~100k-hadith Shia encyclopedia (hubeali).",
    ),
    SourceAdapter(
        slug="tusi",
        corpus=SourceCorpus.TUSI,
        sect=Sect.SHIA,
        acquire_module="tusi",
        parse_module="tusi",
        reachable=True,
        license_note=(
            "narmafraz/ThaqalaynData — the original thaqalayn.net data backend, "
            "CC0 1.0 Universal (public-domain dedication). Carries the real Arabic "
            "source text for the two Books MohammedArab1/ThaqalaynAPI (the 'thaqalayn' "
            "corpus) omits; its non-Arabic translations are AI-generated (pipeline_v4) "
            "and deliberately NOT loaded."
        ),
        description=(
            "ThaqalaynData (CC0) — Tahdhib al-Ahkam + al-Istibsar of al-Tusi, "
            "completing the Shia Four Books (Arabic-only)."
        ),
    ),
)


def _check_registry_integrity() -> None:
    """Fail fast at import on a malformed registry (duplicate slug)."""
    seen: set[str] = set()
    for adapter in SOURCE_REGISTRY:
        if adapter.slug in seen:
            msg = f"duplicate adapter slug in SOURCE_REGISTRY: {adapter.slug!r}"
            raise ValueError(msg)
        seen.add(adapter.slug)


_check_registry_integrity()


def iter_adapters() -> tuple[SourceAdapter, ...]:
    """All registered adapters, in canonical run order."""
    return SOURCE_REGISTRY


def adapter_slugs() -> list[str]:
    """Slugs of all registered adapters, in run order."""
    return [adapter.slug for adapter in SOURCE_REGISTRY]


def get_adapter(slug: str) -> SourceAdapter:
    """Return the adapter registered under *slug*, or raise :class:`KeyError`."""
    for adapter in SOURCE_REGISTRY:
        if adapter.slug == slug:
            return adapter
    msg = f"no adapter registered with slug {slug!r}; known slugs: {adapter_slugs()}"
    raise KeyError(msg)


def adapters_for_corpus(corpus: SourceCorpus) -> list[SourceAdapter]:
    """Adapters whose data lands in *corpus* (usually one; two for ``SUNNAH``)."""
    return [adapter for adapter in SOURCE_REGISTRY if adapter.corpus == corpus]


def adapters_for_sect(sect: Sect) -> list[SourceAdapter]:
    """Adapters that emit records for *sect*.

    Includes adapters that declare *sect* outright, plus multi-sect adapters
    (``sect is None``) that tag ``sect`` per record and so contribute to every
    sect's coverage.
    """
    return [adapter for adapter in SOURCE_REGISTRY if adapter.sect is sect or adapter.sect is None]


def covered_corpora() -> set[SourceCorpus]:
    """The set of :class:`SourceCorpus` values that have at least one adapter."""
    return {adapter.corpus for adapter in SOURCE_REGISTRY}
