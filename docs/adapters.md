# Source adapters — data dictionary

The set of hadith ingest sources is the single registry `SOURCE_REGISTRY` in
[`src/adapters.py`](../src/adapters.py). This document is the human-readable view of
that registry; the registry itself is authoritative (and `tests/test_adapters.py`
fails CI if the two below ever disagree with the code). See
[ADR-002](adr/002-multi-source-adapter-registry.md) for the design rationale.

## The registry contract

Every ingest source is one frozen `SourceAdapter` row declaring:

| Field            | Meaning |
|------------------|---------|
| `slug`           | The adapter key. **Not always the corpus** — see `sunnah` vs `sunnah_scraped` below. |
| `corpus`         | The `SourceCorpus` enum value that namespaces this source's `source_id` (collision-safe across sources — see [`src/parse/identity.py`](../src/parse/identity.py)). Several adapters MAY share one corpus. |
| `sect`           | The declared `Sect` (`sunni` / `shia`), or **`None`** for a multi-sect source that tags `sect` per record in its parser. |
| `acquire_module` / `acquire_fn` | Where the downloader lives under `src/acquire/`. Default fn `run(raw_dir) -> Path \| None`. |
| `parse_module` / `parse_fn`     | Where the parser lives under `src/parse/`. Default fn `run(raw_dir, staging_dir) -> Path \| tuple \| list`. |
| `raw_subdir`     | Per-source raw subdirectory passed to acquire/parse (only `sanadset` uses it). |
| `reachable`      | Whether the source is fetchable in CI today (a `False` source is documented but not currently loadable — e.g. needs an API key). |
| `license_note`   | Licensing / provenance note for the source. |
| `description`    | One-line human description. |

## Registered sources

| slug | corpus | sect | reachable | source |
|------|--------|------|-----------|--------|
| `lk` | `lk` | sunni | yes | LK-Hadith-Corpus (ShathaTm) — 6 Sunni books, open GitHub CSV |
| `sanadset` | `sanadset` | sunni | yes | Sanadset 650K (Mendeley) — 650,986 narrator records / 926 books |
| `thaqalayn` | `thaqalayn` | **shia** | yes | ThaqalaynAPI (MohammedArab1) — the Four Books and more |
| `fawaz` | `fawaz` | multi (`None`) | yes | fawazahmed0/hadith-api — multi-collection, sect per collection |
| `sunnah` | `sunnah` | sunni | **no** | Sunnah.com REST API — gated on an API key (403 keyless, da#71) |
| `sunnah_scraped` | `sunnah` | sunni | yes | Sunnah.com web scraper — keyless; shares the `sunnah` corpus with the API |
| `open_hadith` | `open_hadith` | sunni | yes | Open-Hadith-Data (mhashim6) — 9 Sunni books incl. the Six Books |
| `muhaddithat` | `muhaddithat` | multi (`None`) | yes | muhaddithat/isnad-datasets — female narrators across both traditions |
| `itqan` | `itqan` | multi (`None`) | yes | Itqan rijal DB — 115,735 narrator profiles, 22 classical texts (no upstream license; owner-approved, da#92a) |
| `thaqalayn_data` | `thaqalayn_data` | **shia** | yes | narmafraz/ThaqalaynData (**CC0 1.0**) — Tahdhib al-Ahkam + al-Istibsar of al-Tusi, completing the Four Books (Arabic-only; AI translations omitted, da#182) |

Notes:

- **`sunnah` + `sunnah_scraped` share `SourceCorpus.SUNNAH`** on purpose: the API
  and the scraper describe the same collections, so a shared corpus namespace lets
  the same hadith dedup to one graph node.
- **Multi-sect sources (`sect = None`)** — `fawaz`, `muhaddithat`, `itqan` — span
  both traditions; their parsers set `sect` per record rather than uniformly. They
  contribute to *both* Sunni and Shia coverage in `adapters_for_sect`.
- **`thaqalayn` vs `thaqalayn_data`** are TWO distinct Shia upstreams, kept in
  separate corpora on purpose. `thaqalayn` clones `MohammedArab1/ThaqalaynAPI` (a
  website scrape of thaqalayn.net carrying al-Kafi + al-Faqih of the Four Books);
  `thaqalayn_data` clones `narmafraz/ThaqalaynData` — the original CC0 data backend
  — for the two Books the scrape omits (Tahdhib al-Ahkam + al-Istibsar). Different
  schema and licence, so they never share a parser or corpus namespace. Only the
  genuine Arabic is loaded; ThaqalaynData's non-Arabic translations are
  AI-generated (`verse.ai`, `pipeline_v4`) and deliberately dropped.

## Adding a source

1. Add a distinct value to `SourceCorpus` in [`src/models/enums.py`](../src/models/enums.py)
   — the one shared registration surface, one non-overlapping value per source.
2. Write `src/acquire/<slug>.py` (`run(raw_dir) -> Path | None`) and
   `src/parse/<slug>.py` (`run(raw_dir, staging_dir) -> ...`), normalizing to the
   canonical staging schemas in [`src/parse/schemas.py`](../src/parse/schemas.py)
   and tagging `source_corpus` + `sect`.
3. Add ONE `SourceAdapter` row to `SOURCE_REGISTRY` in `src/adapters.py`.

The coverage invariant in `tests/test_adapters.py` fails CI if step 1 happens
without step 3 (or vice versa), so the enum and the registry cannot drift.
