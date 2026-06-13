# ADR-002: Multi-source adapter registry — one declared contract for ingest sources

## Status: Accepted (P4W4, epic da#81)

## Context

`noorinalabs-data-acquisition` ingests hadith data from nine sources spanning two
traditions (Sunni + Shia). Each source is implemented as a pair of modules — a
downloader in `src/acquire/<slug>.py` (`run(raw_dir) -> Path | None`) and a parser
in `src/parse/<slug>.py` (`run(raw_dir, staging_dir) -> Path | tuple | list`) —
that normalize a disparate source schema into the canonical staging schemas
(`src/parse/schemas.py`), tagged with `source_corpus` and `sect`. The shared
acquisition/parse utilities (`src/acquire/base.py`, `src/parse/base.py`), the
canonical identity grammar (`src/parse/identity.py`, da#82), and the provenance
conformance gate (da#83) were each delivered in their own per-source light-up PRs.

What was missing was a **single source of truth for the set of sources itself.**
Two facts were spread across the codebase and could silently drift:

1. **The source list was duplicated.** `src/acquire/__init__.py` and
   `src/parse/__init__.py` each hand-maintained a parallel `SOURCES` / `PARSERS`
   list of `(name, module)` tuples that had to be kept in lockstep — add a source
   and you had to remember to edit both, in the same order, or the acquire and
   parse phases would disagree on which sources exist.
2. **`sect` was an undeclared, scattered literal.** Each parser hardcoded its sect
   (`SECT = "sunni"`, an inline `"sect": "sunni"`, or — for multi-sect sources —
   tagged it per record). There was no one place that declared *this source is
   Sunni / Shia / multi-sect*, nor a machine-checkable link between that fact and
   the `SourceCorpus` enum that namespaces its ids.

The `SourceCorpus` enum (`src/models/enums.py`) is the one shared registration
surface — every source adds exactly one non-overlapping value — but nothing
enforced that a new enum value actually had a working adapter behind it, or that an
adapter named a real corpus.

## Decision

Introduce **`src/adapters.py`**: a single registry, `SOURCE_REGISTRY`, of frozen
`SourceAdapter` rows. Each row is pure data — it declares the source's `slug`,
`corpus` (`SourceCorpus`), `sect` (`Sect`, or `None` for a multi-sect source that
tags `sect` per record), where its acquire/parse code lives (`acquire_module` /
`parse_module` and the function names), its `reachable` flag, and a `license_note`
+ `description`. The acquire/parse entry points are resolved **lazily** on call
(deferred `import_module`), because `src/adapters.py` is imported *by*
`src/acquire/__init__.py` and `src/parse/__init__.py` and importing their
submodules eagerly would form a package-init cycle.

Consequences of the decision:

- **The two orchestrators derive from the registry.** `src.acquire.run_all` and
  `src.parse.run_all` iterate `SOURCE_REGISTRY` instead of a private list. The
  duplicated `SOURCES` / `PARSERS` lists are deleted. `run_all` signatures are
  unchanged, so `src/cli.py` and every existing test are unaffected.
- **`slug` is the key, not `corpus`.** `sunnah` (the REST API) and `sunnah_scraped`
  (the web scraper) deliberately share `SourceCorpus.SUNNAH` so the same hadith
  dedups to one graph node (`src/parse/identity.py`); the registry therefore keys
  by slug and lets several adapters map to one corpus.
- **A coverage invariant enforces the enum ⇄ registry link.**
  `tests/test_adapters.py` asserts `covered_corpora() == set(SourceCorpus)` — CI
  fails if a `SourceCorpus` value is added without an adapter, or an adapter names
  an unknown corpus. A second test asserts the registry's declared `sect` agrees
  with any parser-level `SECT` constant, so the two cannot diverge.
- **Adding a source is a three-step, mechanically-checked pattern** (see the
  `src/adapters.py` module docstring and `docs/adapters.md`): add the
  `SourceCorpus` value, write the two modules, add one `SourceAdapter` row.

This ADR does **not** re-implement any acquire/parse/identity/schema code that the
per-source PRs already delivered; it ratifies the framework by giving it a single,
declared, enforced registration contract — the capstone of epic da#81.

## Consequences

**Positive**

- One edit site to add/remove a source; acquire and parse can no longer drift.
- `sect`, reachability, and licensing provenance become queryable data
  (`adapters_for_sect`, `get_adapter`, `covered_corpora`) rather than tribal
  knowledge — directly serving da#81's "tag every record with `source_corpus` +
  `sect` consistently" requirement.
- A new `SourceCorpus` value without a working adapter is now a red build, not a
  silent runtime gap.

**Negative / trade-offs**

- The acquire/parse entry points are resolved by string module names rather than
  direct references, so a typo in `acquire_module` surfaces at call time (mitigated
  by `tests/test_adapters.py::TestEntryPoints`, which imports and arity-checks every
  declared entry point without invoking it).
- Parsers still carry their own `SECT` constants for now; the registry is the new
  source of truth and the agreement test guards drift, but a follow-up could have
  the parsers import their sect from the registry to remove the remaining
  duplication entirely.

## Related

- Epic da#81 — multi-source hadith ingestion (Sunni + Shia) with cross-source
  transform/normalization.
- da#82 — canonical `source_id` / identity grammar (`src/parse/identity.py`).
- da#83 — provenance conformance gate (`src/parse/base.py::provenance_violations`).
- `docs/adapters.md` — the per-source data dictionary generated from this registry's
  shape.
