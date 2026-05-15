# Implementation Plan — Issue #2

## Metadata
- **Issue**: #2 — Restructure repo: best practices for app+infra monorepo, tests, venvs, security
- **Branch**: `feat/restructure-repo-*` (one branch per batch)
- **Date**: 2026-05-15

## Status

| Step | Status | Notes |
|------|--------|-------|
| Context gathered | done | |
| Plan approved | pending | |
| Implementation batch 1 | pending | Steps 1-3 |
| Implementation batch 2 | pending | Steps 4-8 |
| Implementation batch 3 | pending | Steps 9-10 |

## Problem Statement

Repo has flat `app/` dir with 2 Python files, no tests, dual dependency specs, no linting, no CI. Need modern Python project structure following best practices.

## Non-Goals

- **Do NOT touch `infra/`** — Terraform modules stay as-is
- **Do NOT modify ChromaDB data/schema** — EC2 database untouched
- **Do NOT change runtime behavior** — scraper pipeline must work identically after restructure
- **Do NOT change AWS Secrets Manager config** — secrets access pattern stays same

## Safety Constraints

- `infra/` directory: read-only, no modifications
- ChromaDB connection logic: preserve exact same HTTP client config
- Docker image must still run cron job identically
- All environment variables and secret names unchanged

## Current State

```
app/
├── __init__.py          # empty
├── main.py              # 448 lines — monolith orchestrator
├── utils.py             # 481 lines — all helper functions
├── Dockerfile           # python:3.10-slim + cron
├── requirements.txt     # 7 deps, no pins
├── environment.yml      # conda spec, says python 3.11
├── new.ipynb            # exploration notebook
├── db.ipynb             # DB notebook
└── README.md
```

## Batch Breakdown

### Batch 1 — Structural Foundation (Issue #X)
**Branch**: `feat/restructure-repo-layout`

| File | Change |
|------|--------|
| `pyproject.toml` | NEW — deps, pytest, ruff, mypy config |
| `src/xflats/__init__.py` | NEW — package init |
| `src/xflats/main.py` | MOVE from `app/main.py` — update imports |
| `src/xflats/scraper/__init__.py` | NEW |
| `src/xflats/scraper/boligsiden.py` | NEW — extract scraping logic from main.py |
| `src/xflats/extraction/__init__.py` | NEW |
| `src/xflats/extraction/gemini.py` | NEW — extract AI logic from main.py+utils.py |
| `src/xflats/storage/__init__.py` | NEW |
| `src/xflats/storage/chromadb.py` | NEW — extract ChromaDB logic from main.py+utils.py |
| `src/xflats/notifications/__init__.py` | NEW |
| `src/xflats/notifications/telegram.py` | NEW — extract Telegram logic from main.py |
| `src/xflats/config/__init__.py` | NEW |
| `src/xflats/config/secrets.py` | NEW — extract Config class + get_secret from main.py+utils.py |
| `src/xflats/utils.py` | NEW — shared helpers (geocoding, HTML processing) |
| `docker/Dockerfile` | MOVE from `app/Dockerfile` — update paths for src layout |
| `docker/crontab` | NEW — extract cron schedule from Dockerfile |
| `notebooks/new.ipynb` | MOVE from `app/new.ipynb` |
| `notebooks/db.ipynb` | MOVE from `app/db.ipynb` |
| `app/` | DELETE directory after moves complete |

**Key risk**: Import paths change. Must verify `python -m xflats.main` works.

### Batch 2 — Tests & Tooling (Issue #Y)
**Branch**: `feat/restructure-repo-tests`

| Item | Detail |
|------|--------|
| `tests/conftest.py` | Shared fixtures, mock ChromaDB/Gemini/Telegram |
| `tests/unit/test_scraper.py` | Test URL extraction, HTML preprocessing |
| `tests/unit/test_extraction.py` | Test Gemini prompt formatting, response parsing |
| `tests/unit/test_storage.py` | Test ChromaDB add/query with mock client |
| `tests/unit/test_notifications.py` | Test Telegram message formatting |
| `Makefile` | test, lint, docker, tf-plan targets |
| `.pre-commit-config.yaml` | ruff + detect-secrets hooks |
| `.github/workflows/ci.yml` | Lint + test on PR |
| `.github/workflows/deploy.yml` | Docker build + ECR push on main |

### Batch 3 — Cleanup (Issue #Z)
**Branch**: `feat/restructure-repo-cleanup`

| Item | Detail |
|------|--------|
| Remove `requirements.txt` | Replaced by pyproject.toml |
| Remove `environment.yml` | Replaced by pyproject.toml |
| Update `README.md` | New structure, dev setup, make targets |
| Add `docs/architecture.md` | System overview diagram |
| Update `.gitignore` | Add .env*, .venv, dist/ |

## Verification

- [ ] `python -c "from xflats.main import RealEstateScraper"` works
- [ ] `python -m py_compile` passes on all new .py files
- [ ] Docker build succeeds with new layout
- [ ] `ruff check src/` passes
- [ ] Existing infra/ completely untouched (git diff shows zero changes)
- [ ] No secrets in committed files

## Risks & Open Questions

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Import path changes break Docker cron | Test Docker build + manual run before PR |
| 2 | ChromaDB connection changes | Keep exact same HTTP client config, just move to new file |
| 3 | Notebooks may have hardcoded imports | Update notebook imports or mark as needs-update |
| 4 | `osmnx` has heavy deps (geopandas) | Keep in deps, may explore lighter alternative later |

## Issues Log

| # | Issue | Resolution |
|---|-------|------------|
| | | |
