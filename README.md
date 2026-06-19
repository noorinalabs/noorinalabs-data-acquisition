# noorinalabs-data-acquisition

Data source acquisition — scrapers, API connectors, downloaders; Python + PyArrow.

This repo is the **acquisition** stage of the NoorinALabs data pipeline: it pulls
raw hadith/rijal source material from upstream collections (web scrapers and API
connectors), normalizes it into a stable on-disk form, and writes typed columnar
artifacts (PyArrow/Parquet) for the downstream ingest platform to consume. Source
code lives under `src/`, tests under `tests/`.

## Git hooks (required)

This repo mirrors its CI checks locally via [pre-commit](https://pre-commit.com/).
After cloning, install BOTH hook stages once:

```bash
pre-commit install                       # commit-stage checks
pre-commit install --hook-type pre-push  # push-stage checks
```

- **Commit stage** runs: `ruff-format`, `ruff-lint` (with `--fix`), `gitleaks`
  secret-scan, and `actionlint` over the workflows.
- **Pre-push stage** runs: `mypy` (strict, over `src/`) and the `pytest` unit
  suite (`tests/`, with `ENVIRONMENT=test`, excluding integration tests).

These mirror `.github/workflows/ci.yml` so failures surface locally before a PR
(org-wide local⇄CI parity, noorinalabs-main#684). Never bypass with `--no-verify`.
If `pre-commit install` "cowardly refuses" because `core.hooksPath` is set, run
`git config --unset core.hooksPath` first.

Documentation and config (markdown lint, cspell spellcheck, lychee link-check,
YAML/JSON syntax, and a pinned `actionlint`) are additionally gated in CI by
`.github/workflows/docs.yml`.
