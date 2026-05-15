# Implementation Plan — Issue #8

## Metadata
- **Issue**: #8 — chore: upgrade Python to 3.14
- **Branch**: `chore/8/upgrade-python-3.14`
- **Date**: 2026-05-15

## Status

| Step | Status | Notes |
|------|--------|-------|
| Context gathered | done | |
| Plan approved | done | |
| Implementation | done | |
| Tested | done | 21 tests pass, ruff clean, import OK |
| PR created | pending | |

## Problem Statement

Project targets Python 3.11. Python 3.14 stable since Oct 2025 (now 3.14.5). Upgrade for performance, better errors, new stdlib. Fallback to 3.13 if deps break.

## Non-Goals

- No new Python 3.14 features in app code (just version bump)
- No CI/CD changes (separate issue)

## Current State

| File | Current | Target |
|------|---------|--------|
| `pyproject.toml` requires-python | `>=3.11` | `>=3.14` |
| `pyproject.toml` ruff target-version | `py311` | `py314` |
| `pyproject.toml` mypy python_version | `3.11` | `3.14` |
| `docker/Dockerfile` base image | `python:3.11-slim` | `python:3.14-slim` |

## Proposed Changes

### Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Update requires-python, ruff target-version, mypy python_version |
| `docker/Dockerfile` | Update base image to python:3.14-slim |
| `uv.lock` | Regenerate with Python 3.14 |
| `README.md` | Update Python version reference if present |

### Implementation Details

1. Update pyproject.toml — 3 version strings
2. Update Dockerfile base image
3. Install Python 3.14 via uv: `uv python install 3.14`
4. Regenerate lock: `uv lock`
5. Sync deps: `uv sync --dev`
6. Run ruff: `uv run ruff check src/`
7. Run import test: `uv run python -c "from xflats.main import RealEstateScraper"`
8. Run tests: `uv run pytest`
9. Docker build test (ask user)

If any dep fails on 3.14, fallback all versions to 3.13.

## Verification

- [ ] `uv run python --version` shows 3.14.x
- [ ] `uv run ruff check src/` passes
- [ ] `uv run pytest` passes
- [ ] Docker build succeeds
- [ ] Import test passes

## Risks & Open Questions

- chromadb C extensions may not support 3.14 yet — fallback to 3.13
- osmnx optional dep may have issues
- ruff may not recognize `py314` target — will check

## Issues Log

| # | Issue | Resolution |
|---|-------|------------|
| | | |
