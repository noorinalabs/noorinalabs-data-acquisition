# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**noorinalabs-isnad-graph-ingestion** is the data ingestion pipeline for the isnad-graph platform. It acquires hadith data from 8+ sources, parses and normalizes it into Parquet staging files, performs entity resolution (narrator NER, disambiguation, deduplication), loads data into Neo4j and PostgreSQL, and enriches the graph with metrics and classifications.

This repository was extracted from `noorinalabs-isnad-graph` (which contained all pipeline code in `src/acquire/`, `src/parse/`, `src/resolve/`, `src/graph/`, `src/enrich/`, `src/models/`, `src/utils/`). **Phase 1 is the extraction itself.**

## Tech Stack

- **Python 3.14** with **uv** as the package manager
- **PyArrow / Parquet** — columnar staging format with schema enforcement
- **Neo4j 5.x** — target graph database (narrator networks, isnad chains)
- **PostgreSQL 16+ with pgvector** — relational metadata, vector embeddings
- **CAMeLBERT** — Arabic-specific transformer for NER and text processing
- **FAISS** — CPU-optimized vector similarity for entity deduplication
- **sentence-transformers** — embedding generation for semantic similarity
- **Pydantic v2** — frozen data models with strict validation
- **httpx** — async HTTP client for API acquisition
- **ruff** — linting and formatting
- **mypy** — strict type checking
- **pytest** — testing framework

## Architecture

### Pipeline Stages

| Module | Phase | Purpose |
|--------|-------|---------|
| `src/models/` | 0 | Frozen Pydantic v2 models for all graph nodes and edges |
| `src/utils/` | 0 | Arabic text processing, Neo4j/PG clients, structured logging |
| `src/config.py` | 0 | Pydantic Settings (loads `.env`), singleton via `get_settings()` |
| `src/acquire/` | 1 | Downloaders for 8 data sources → `data/raw/` |
| `src/parse/` | 1 | Parsers producing normalized Parquet → `data/staging/` |
| `src/validate/` | 1-4 | Data quality validation and profiling |
| `src/resolve/` | 2 | Narrator NER/disambiguation, hadith dedup (CAMeLBERT, FAISS) |
| `src/graph/` | 3 | Neo4j node/edge loaders, validation queries |
| `src/enrich/` | 4 | Graph metrics (centrality, PageRank, Louvain), topic classification, historical overlay |

### Data Flow

```
Raw sources (CSV, JSON, REST APIs)
  → data/raw/           (acquire phase)
  → data/staging/        (parse phase, Parquet)
  → entity resolution    (resolve phase)
  → Neo4j graph          (graph load phase)
  → enriched graph       (enrich phase)
```

### Data Sources

1. **Sunnah.com API** — REST API for Sunni hadith collections
2. **Kaggle datasets** — Pre-compiled hadith datasets
3. **GitHub repos** — Open-source hadith data repositories
4. **Al-Bukhari CSV** — Sahih al-Bukhari collection
5. **Muslim CSV** — Sahih Muslim collection
6. **Abu Dawud** — Sunan Abu Dawud
7. **Shia hadith sources** — Al-Kafi and related collections
8. **Narrator biographical databases** — Rijal/biographical dictionaries

## Build & Development Commands

```bash
# Setup
make setup              # Install Python dependencies with uv
make setup-hooks        # Configure git hooks

# Full pipeline
make pipeline           # Run full ETL pipeline end-to-end

# Individual stages
make acquire            # Phase 1: download raw data
make parse              # Phase 1: parse to staging Parquet
make resolve            # Phase 2: entity resolution (NER + disambiguation + dedup)
make load               # Phase 3: load into Neo4j
make enrich             # Phase 4: compute metrics, topics, historical overlay

# Data validation
make validate           # Run data quality validation (strict mode, JSON report)
make validate-staging   # Validate staging Parquet files (warn mode)
make profile-data       # Profile staging Parquet files

# Quality
make test               # Run pytest suite
make lint               # Run ruff linter
make format             # Run ruff formatter
make typecheck          # Run mypy (strict mode)
make check              # Run all CI checks (lint + typecheck + test)

# Cleanup
make clean              # Remove staging data and caches
```

## Code Conventions

- **Ruff** for linting and formatting, line length 100
- **mypy** strict mode with pydantic plugin
- All Pydantic models use `ConfigDict(frozen=True)` for immutability
- All enums are `(str, Enum)` for clean JSON/Parquet serialization
- All downloaders and loaders must be **idempotent** (safe to re-run)
- Arabic text utilities are pure Python: diacritics stripping, alif/hamza/taa marbuta normalization
- Staging data uses PyArrow schemas as intermediate between raw data and graph nodes

## Configuration

Copy `.env.example` to `.env`. Key variables:

**Databases:**
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `PG_DSN`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`

**APIs:**
- `SUNNAH_API_KEY` — required for Sunnah.com API
- `KAGGLE_USERNAME`, `KAGGLE_KEY` — required for Kaggle datasets

**Application:**
- `DATA_RAW_DIR`, `DATA_STAGING_DIR`, `DATA_CURATED_DIR` — data paths
- `LOG_LEVEL`, `LOG_FORMAT` — logging configuration

## Project Memory

Project memory is **version-controlled in this repo** at `.claude/memory/`, not in the user-space auto-memory directory. This makes the accumulated state **transferable**: a developer who pulls a branch gets the memory with it, with zero per-machine setup. The index below is auto-loaded into every session via this committed import:

@.claude/memory/MEMORY.md

`MEMORY.md` is the always-loaded index (one line per memory); the individual topic files in `.claude/memory/*.md` are read on demand when a line looks relevant. This repo is **self-contained**: it imports only its own `.claude/memory/` and never across repos. The corpus here is the data-acquisition-specific memory split out of `noorinalabs-main` (org/repo split — meta noorinalabs-main#740, seed da#203); a few `[[wikilinks]]` still point at org-level memories that remain in the parent and may dangle — cross-repo soft pointers are acceptable.

**Recording a memory:** create or edit `.claude/memory/<kebab-slug>.md` with the standard frontmatter (`name`, `description`, `metadata.type` = `user` | `feedback` | `project` | `reference`), add a one-line pointer to `MEMORY.md` (`- [Title](file.md) — hook`), and **commit it** so it travels with the branch. Link related memories with `[[other-slug]]`. Before adding, check for an existing file covering the same fact and update it instead of duplicating; delete memories that turn out to be wrong.

> `.claude/memory/**` is excluded from the markdown/cspell/lychee linters (dense append-only note prose with names, SHAs, `[[wikilinks]]`, and Arabic) — the exclusions live in `.markdownlint-cli2.jsonc`, `.cspell.json`, and `.lychee.toml`.

## Team Workflow

> **Cross-repo session-team note:** The team structure described below is the **per-repo team** — operative when a session is opened isolated in this repo for repo-only work.
>
> When work is orchestrated from the parent `noorinalabs-main` (the common case — wave kickoff, cross-repo features, wave-coordinated bug fixes), all spawned agents — regardless of which repo they edit — join the single `noorinalabs` session team. The per-repo roster below still governs **commit identity, domain ownership, and reviewer pairing**, but the team-creation surface lives in the orchestrator session, not here.
>
> See `noorinalabs-main/CLAUDE.md` § "Session team architecture" and `noorinalabs-main/.claude/team/charter/agents.md` § "Single-Leader Constraint" for the delegation pattern.

**All work MUST be executed through the simulated team structure.** No work begins without spawning the team.

> **Note:** The authoritative team config (charter, roster, hooks, skills) lives in the parent repo (`noorinalabs-main/.claude/`). This repo retains a local copy for agents working within noorinalabs-isnad-graph-ingestion.

- **Charter & rules:** `.claude/team/charter.md` (canonical: `../../.claude/team/charter.md`)
- **Active roster:** `.claude/team/roster/` (canonical: `../../.claude/team/roster/`)
- **Feedback log:** `.claude/team/feedback_log.md`

### Team Composition
| Role | Level | Name | File |
|------|-------|------|------|
| Pipeline Manager | Senior VP (Executive) | Dilara Erdogan | `roster/manager_dilara.md` |
| Data Architect | Partner | Jean-Claude Habimana | `roster/architect_jeanclaude.md` |
| Data Engineer | Staff | Alejandra Reyes-Fuentes | `roster/data_engineer_alejandra.md` |
| Data Engineer | Senior | Kavitha Sundaramurthy | `roster/data_engineer_kavitha.md` |
| Data Engineer | Senior | Nikolaos Papadopoulos | `roster/data_engineer_nikolaos.md` |
| ML/NLP Engineer | Senior | Ivana Horvat | `roster/ml_engineer_ivana.md` |
| Integration Engineer | Senior | Kwesi Boateng | `roster/integration_engineer_kwesi.md` |
| QA / Data Quality Engineer | Senior | Oyunbileg Batbayar | `roster/qa_engineer_oyunbileg.md` |
| DevOps Engineer | Senior | Tarek Mansour | `roster/devops_engineer_tarek.md` |
| Technical Writer | Senior | Sofia Cardoso | `roster/tech_writer_sofia.md` |

### Key Rules
- **Commit identity:** Each team member commits using per-commit `-c` flags with their name and `parametrization+{FirstName}.{LastName}@gmail.com` email — **never** set global/repo git config. See `.claude/team/charter.md` § Commit Identity for the full table.
- **Worktrees** are the preferred isolation method for all code-writing agents
- Manager spawns team members, creates stories/AC from PRD, and owns timelines
- Manager, Data Architect, and DevOps Engineer coordinate to prevent cross-team blocking
- Feedback flows up and down; severe feedback triggers fire-and-replace
- If the Manager receives significant negative feedback from the user, the Manager is replaced
- Team evolves toward steady state of minimal negative feedback

## Developer Tooling & Orchestration

- **gh-cli** is installed and available from the terminal
- **SSH access** is enabled from the terminal
- **GitHub Projects** — project/feature tracking and board management
- **GitHub Issues** — story/task/bug tracking (created by Manager, assigned to team members)
- **GitHub Actions** — CI/CD pipelines, automated tests, linting, deployment
- These three (Projects, Issues, Actions) are the **core orchestration layer** — do not introduce alternative tools for these concerns
- **Branching strategy:** Feature branches named `{FirstInitial}.{LastName}/{IIII}-{issue-name}` (e.g., `D.Erdogan/0001-extract-acquire-module`) merged to `main` via PR

## Phase 1: Extraction from noorinalabs-isnad-graph

The initial work is extracting pipeline code from the monorepo:
1. Copy `src/acquire/`, `src/parse/`, `src/resolve/`, `src/graph/`, `src/enrich/`, `src/models/`, `src/utils/` from `noorinalabs-isnad-graph`
2. Extract pipeline-specific dependencies from `pyproject.toml`
3. Create standalone `pyproject.toml` with uv
4. Set up CI/CD (GitHub Actions for lint, typecheck, test)
5. Verify all pipeline commands work independently
6. Update import paths and cross-repo interfaces
