# Operational Runbook — noorinalabs-data-acquisition

This runbook is the on-call reference for operating the Acquire stage of the
NoorinALabs hadith data pipeline. It covers local build/run, source connector
configuration, B2 / MinIO data-pipeline procedures, rollback, common-failure
triage, and on-call escalation.

For repo-wide context (architecture, phases, source list) see `CLAUDE.md`. For
org-wide pipeline conventions see `noorinalabs-main/ontology/conventions.md`.

## Scope

This repo is the **Acquire stage only** — scrapers, REST connectors, and Git
or Kaggle downloaders. Output lands in B2 under
`noorinalabs-pipeline/raw/{source}/{date}/`, and a `pipeline.raw.landed` Kafka
signal is emitted per file. Downstream stages (dedup, enrich, normalize,
graph-load) live in `noorinalabs-isnad-ingest-platform`. Parquet parsing,
entity resolution, Neo4j load, and graph enrichment are kept here in
`src/parse/`, `src/resolve/`, `src/graph/`, `src/enrich/` for the legacy
all-in-one CLI path; the canonical flow is acquire → B2 → Kafka → platform.

| Stage | Module | Output |
|-------|--------|--------|
| Acquire (canonical) | `src/acquire/` | `data/raw/` then B2 `raw/{source}/{date}/` |
| Parse (legacy CLI) | `src/parse/` | `data/staging/*.parquet` |
| Resolve (legacy CLI) | `src/resolve/` | `data/curated/*.parquet` |
| Graph load (legacy CLI) | `src/graph/` | Neo4j |
| Enrich (legacy CLI) | `src/enrich/` | Neo4j metrics, PG vectors |

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `make setup` |
| Configure pre-commit hooks | `make setup-hooks` |
| Show config + DB connectivity | `uv run isnad-ingest info` |
| Run acquire only | `make acquire` |
| Run full legacy pipeline | `make pipeline` |
| Validate staging Parquet | `make validate-staging` |
| Lint / format / typecheck | `make lint` / `make format` / `make typecheck` |
| Full CI checks | `make check` |
| Reset staging + caches | `make clean` |

## Build & Run

### Prerequisites

- Python 3.14 (`pyproject.toml` requires `>=3.14`); ruff target is `py313`
  because ruff has not shipped 3.14 grammar yet.
- `uv` package manager.
- For graph load (`make load`) and enrich (`make enrich`): a reachable Neo4j 5.x
  and PostgreSQL 16+ instance.
- For acquire stage: outbound HTTPS, a populated `.env`, and (optionally) a
  reachable B2 endpoint plus Kafka broker. Both are no-op-on-missing; see
  Configuration below.
- For B2 upload from a local connector run, valid B2 application-key
  credentials with write access to the `noorinalabs-pipeline` bucket. See
  `noorinalabs-main` memory `project_data_pipeline_architecture` for the
  storage-architecture rationale.

### Setup

```bash
cp .env.example .env       # then fill in values — see Configuration below
make setup                 # uv sync --group ml (installs ML extras for resolve)
make setup-hooks           # pre-commit hooks (ruff, mypy)
uv run isnad-ingest info   # verify config + DB connectivity
```

`uv sync --group ml` pulls heavyweight ML deps (sentence-transformers, faiss,
torch, camel-tools). If the box only needs to run acquire, drop the group:
`uv sync` is sufficient and significantly faster.

### Acquire

```bash
make acquire                            # all sources
uv run isnad-ingest acquire --source=sunnah_api   # single source (if supported)
```

Connector list (see `src/acquire/`):

| Connector | Source | Auth | Output |
|-----------|--------|------|--------|
| `sunnah_api` | sunnah.com REST API | `SUNNAH_API_KEY` | `data/raw/sunnah_api/<date>/` |
| `sunnah_scraper` | sunnah.com HTML fallback | none | `data/raw/sunnah_scraper/<date>/` |
| `lk_corpus` | LK Hadith corpus (Git) | none | `data/raw/lk_corpus/<date>/` |
| `fawaz` | Fawaz hadith dataset (Git) | none | `data/raw/fawaz/<date>/` |
| `open_hadith` | OpenHadith repos | none | `data/raw/open_hadith/<date>/` |
| `sanadset` | Sanadset Kaggle dataset | `KAGGLE_USERNAME`, `KAGGLE_KEY` | `data/raw/sanadset/<date>/` |
| `muhaddithat` | Muhaddithat dataset | none | `data/raw/muhaddithat/<date>/` |
| `thaqalayn` | Thaqalayn Shia hadith | none | `data/raw/thaqalayn/<date>/` |

Acquire is **idempotent**: if `dest.exists() and dest.stat().st_size > 0` the
connector skips download (`src/acquire/base.py::download_file`). Pass
`overwrite=True` (or use `make clean` first) for forced re-pull. Per-connector
retry is handled by `tenacity` (`stop_after_attempt(3)`,
`wait_exponential`).

### Legacy end-to-end CLI

```bash
make pipeline   # acquire → parse → resolve → load → enrich
```

`make pipeline` runs sequentially via the `isnad-ingest` CLI; failure aborts
on the failing stage. For the Kafka-driven canonical path, only run
`make acquire`; downstream consumers in
`noorinalabs-isnad-ingest-platform` pick up the `pipeline.raw.landed` signal.

## Configuration

`.env` is loaded by Pydantic Settings (`src/config.py`). All variables are
documented in `.env.example`; the table below covers what on-call typically
needs to touch.

| Variable | Required for | Default | Notes |
|----------|--------------|---------|-------|
| `NEO4J_URI` | `load`, `enrich` | `bolt://localhost:7687` | |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `load`, `enrich` | `neo4j` / `isnad_graph_dev` | |
| `PG_DSN` | `load`, `enrich` | `postgresql://isnad:isnad_dev@localhost:5432/isnad_graph` | |
| `SUNNAH_API_KEY` | `acquire/sunnah_api` | `""` | Empty → connector errors on first call |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | `acquire/sanadset` | `""` | |
| `DATA_RAW_DIR` | acquire | `./data/raw` | |
| `DATA_STAGING_DIR` | parse | `./data/staging` | |
| `DATA_CURATED_DIR` | resolve | `./data/curated` | |
| `LOG_LEVEL` / `LOG_FORMAT` | always | `INFO` / `console` | structlog; set `json` in prod |
| `KAFKA_BOOTSTRAP_SERVERS` | acquire emit | unset | **Unset → producer is a no-op**, messages are debug-logged and dropped (`src/messaging/kafka_producer.py`) |
| `KAFKA_RAW_LANDED_TOPIC` | acquire emit | `pipeline.raw.landed` | Must match consumer topic in ingest-platform |

> **B2 / MinIO:** the canonical acquire emits `pipeline.raw.landed` whose
> `b2_key` references `raw/<source>/<date>/<filename>`. The actual upload
> path (B2 SDK, S3 endpoint, MinIO local) lives in deploy infrastructure;
> source-side env-var conventions follow the org pattern documented in
> `noorinalabs-main` memory `project_data_pipeline_architecture` and in
> `noorinalabs-deploy` (canonical bootstrap runbook). Do not invent local
> bucket names — coordinate with the DevOps on-call before changing prefix
> or bucket conventions.

### Local-dev MinIO

Local development substitutes MinIO for B2 over the same S3-compatible code
path. Bring up MinIO from `noorinalabs-deploy`'s compose stack, point the B2
endpoint env var to the MinIO host, and use `noorinalabs-pipeline` as the
bucket name to match prod prefixing. See `noorinalabs-deploy/RUNBOOK.md` (when
landed under deploy#24) for the compose-up procedure and credential setup.

## Verification

After an acquire run:

1. **Disk**: `data/raw/<source>/<date>/` is non-empty and the connector log
   shows `download_skipped` only for re-runs of the same date.
2. **Manifest**: `find data/raw -name '.manifest.json'` is up to date for
   each source (manifest helpers live in `src/pipeline/manifest.py`).
3. **B2 / MinIO** (when configured): list the bucket prefix
   `noorinalabs-pipeline/raw/<source>/<YYYY-MM-DD>/` and check object count
   matches local file count.
4. **Kafka** (when configured): `pipeline.raw.landed` lag drops to 0 after
   downstream consumers process the batch. The producer is fire-and-forget
   and flushes on process exit; if the acquire process was `kill -9`'d
   without flush, replay from B2 (see Rollback).

For a successful legacy `make pipeline`:

```bash
make validate          # strict-mode validation, report at data/reports/validation_report.json
make profile-data      # PyArrow profile of staging Parquet
```

## Common Failures & Triage

### Acquire — HTTP rate limit (429 / 503)

**Symptom**: connector logs `httpx.HTTPStatusError` 429 or 503 from
`sunnah_api` / `sunnah_scraper`; tenacity exhausts after 3 attempts.

**Triage**:
1. Check the source's published rate-limit policy. For `sunnah_api`, log into
   the developer portal and inspect quota.
2. `make clean` is **not** required — acquire is idempotent.
3. Re-run `make acquire` after a 5–15 minute cooldown.
4. If recurring, lower per-connector concurrency (single-source mode) and
   file a tracking issue.

**Do not** add a synthetic `sleep` to bypass rate-limits in code without
discussion — the right fix lives in connector-level backoff.

### Acquire — auth failure

**Symptom**: `sunnah_api` connector raises 401/403 immediately; `sanadset`
connector raises Kaggle auth error.

**Triage**:
1. `uv run isnad-ingest info` to confirm the env var is loaded (passwords are
   masked but presence/absence is visible).
2. Re-issue the API key from the source's developer portal; rotate the
   secret in the deployment-side secret store (see
   `noorinalabs-deploy/RUNBOOK.md`).
3. Update `.env` (local) or the secret in prod, redeploy or restart, re-run.

### Acquire — B2 upload failure

**Symptom**: connector completes local download but the B2 client raises
`AccessDenied` / `NoSuchBucket` / network timeout.

**Triage**:
1. Confirm the application-key has `writeFiles` for the
   `noorinalabs-pipeline` bucket.
2. Confirm the endpoint URL matches the region of the bucket (B2 endpoints
   are region-pinned).
3. For a single bad object, re-run the acquire — `download_file` skips
   already-downloaded local files and the upload helper is idempotent on
   identical content.
4. For a wholesale bucket-side issue, page the DevOps on-call (escalation
   below) — credential rotation or bucket-policy fix is owned there.

### Acquire — Kafka emit failure

**Symptom**: connector logs `kafka_emit_failed` for one or more files but the
acquire process completes successfully.

**Triage**:
1. Per design, Kafka emit is best-effort and **must not fail the acquire**
   (see `src/messaging/kafka_producer.py` module docstring). Local files and
   B2 uploads are still durable.
2. Reconcile by replaying from B2 — list `noorinalabs-pipeline/raw/<source>/`
   and re-emit the missed keys via the platform-side replay tool (lives in
   `noorinalabs-isnad-ingest-platform`).
3. If the broker is genuinely down, page the platform on-call and pause
   further acquires until brokers are healthy (avoids a backlog of missed
   emits).

### Parse — schema validation error

**Symptom**: `make parse` aborts with a Pydantic / PyArrow schema mismatch;
`data/staging/` is partial.

**Triage**:
1. Identify the offending source from the traceback.
2. Inspect the raw file under `data/raw/<source>/<date>/` for shape changes
   (extra/missing columns, encoding issues).
3. If the upstream changed, file a parser-update issue and tag the source
   maintainer; do **not** silently widen the schema in `src/parse/schemas.py`
   without review.

### Resolve — torch / faiss / camel-tools import error

**Symptom**: `make resolve` fails with `ModuleNotFoundError`.

**Triage**: the ML group was not installed. Run `make setup` (which uses
`uv sync --group ml`) on the box. If disk pressure prevents the install,
acquire-only operation does not require this group.

### Load — Neo4j / Postgres unreachable

**Symptom**: `uv run isnad-ingest info` shows `neo4j: unavailable` or
`postgres: unavailable`.

**Triage**:
1. Confirm the DB process is up and the URI/DSN host is reachable from the
   box.
2. Check credentials match the deployed values (rotation lives in
   `noorinalabs-deploy`).
3. For a slow Neo4j startup (cold container), wait 30–60s and re-check.

## Rollback

The Acquire stage is **append-only by date**. There is no in-place update to
roll back. Rollback shapes by surface:

### Rollback a bad connector change

If a code change in `src/acquire/<connector>.py` produces malformed output:

1. **Disable the connector** by either reverting the offending commit
   (preferred) or short-circuiting the connector in `src/acquire/__init__.py`
   if the bad code is on `main` and a hotfix revert PR is already in flight.
2. Optionally remove the bad date partition: `rm -rf data/raw/<source>/<date>/`
   locally and `b2 rm --recursive b2://noorinalabs-pipeline/raw/<source>/<date>/`
   server-side. **Coordinate the B2 deletion with DevOps on-call** —
   downstream consumers may have already processed the data and a deletion
   without a corresponding consumer-side reset will desync the pipeline.
3. Re-run acquire after the revert lands.

### Rollback a connector-config change

For credential / endpoint / topic env-var rollback, revert the secret in the
deployment store and restart. Do **not** roll back env-var changes by editing
`.env` in a running container without re-loading — Pydantic Settings is
cached via `lru_cache` (`get_settings()`); a restart is required.

### Rollback a B2 / MinIO storage migration

Out of scope for this repo — owned by `noorinalabs-deploy`. Page the DevOps
on-call.

### Disable a scraper

Short-term disable (e.g., source is offline, throwing 503 in a loop):

1. Edit `src/acquire/__init__.py` and remove the connector from the registry,
   OR comment out its entry in the per-connector loop in
   `acquire/run_all`.
2. Land the change as a small PR labelled `ops` (not a wave issue) so the
   audit trail is clean.
3. Re-enable by reverting the patch when upstream recovers.

## On-Call Escalation

| Role | When to page | Where |
|------|--------------|-------|
| Pipeline Manager (Dilara Erdogan) | Acquire failures lasting > 1 hour, data-quality regression | data-acquisition team channel |
| DevOps Engineer (Tarek Mansour) | B2 / MinIO / Kafka infra issues, secret rotation | data-acquisition team channel; cross-page deploy on-call if infra-wide |
| Program Director (Nadia Khoury) | Cross-repo blocking, missed wave commitments | noorinalabs org channel |
| Standards & Quality Lead (Aino Virtanen) | Convention drift, hook bypass, charter violations | noorinalabs org channel |

For B2/Kafka outages spanning the whole pipeline, page the deploy on-call
first — this repo's runbook covers application-side triage; infra owns the
substrate.

## Cross-References

- `noorinalabs-data-acquisition/CLAUDE.md` — repo overview, tech stack,
  team composition.
- `noorinalabs-main/ontology/conventions.md` — pipeline-stage definitions
  (Acquire / Parse / Resolve / Load / Enrich).
- `noorinalabs-main/ontology/services.yaml` — `data-acquisition` service
  entry, including `output_target` and `local_dev` mapping.
- `noorinalabs-isnad-ingest-platform` — downstream `pipeline.raw.landed`
  consumers, dedup / enrich / normalize / graph-load workers.
- `noorinalabs-deploy/RUNBOOK.md` — compose stack, MinIO, B2 secret
  rotation (Tier-2 sibling, deploy#24).

## Maintenance

This runbook lives next to the code; update it in the same PR as any change
to acquire behaviour, environment variables, source list, B2/Kafka contract,
or rollback procedure. Drift between this runbook and reality is an on-call
trap — fix it inline, not in a follow-up.
