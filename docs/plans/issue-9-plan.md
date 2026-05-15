# Implementation Plan — Issue #9

## Metadata
- **Issue**: #9 — chore: adopt Google Python Style Guide across codebase
- **Branch**: `chore/9/google-style-guide`
- **Date**: 2026-05-15

## Status

| Step | Status | Notes |
|------|--------|-------|
| Context gathered | done | All 7 source files analyzed |
| Plan approved | pending | |
| Implementation | pending | |
| Tested | pending | |
| PR created | pending | |

## Problem Statement

Codebase lacks consistent style — missing docstrings (~18 functions), partial type annotations, 24 `print()` calls instead of logging, 4 broad `except Exception` blocks, no docstring linting. Issue #9 asks to align with Google Python Style Guide.

## Non-Goals

- No behavioral changes — style/tooling only
- No new features or logic changes
- Not changing line-length from 88 (ruff default, close enough to Google's 80 — pragmatic choice)
- Not refactoring architecture or moving files

## Current State

| Issue | Count | Files |
|-------|-------|-------|
| Missing docstrings | ~18 funcs | all source files |
| print() calls | 24 | main.py(13), chromadb.py(4), telegram.py(4), utils.py(2), boligsiden.py(1) |
| Broad except Exception | 4 | main.py(3), boligsiden.py(1) |
| Missing type annotations | ~8 funcs | secrets.py(2 fully untyped), chromadb.py(untyped `collection` params), telegram.py |
| Bare `dict` types | 7 | chromadb.py(4), telegram.py(1), utils.py(1), gemini.py(1) |
| Ruff D rules | 0 | not configured |

## Proposed Changes

### Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add ruff D, T20, BLE rules + pydocstyle google convention |
| `src/xflats/main.py` | Add logging, docstrings, fix 3 broad excepts |
| `src/xflats/utils.py` | Add logging, docstrings, fix bare dict types |
| `src/xflats/config/secrets.py` | Add docstrings, full type annotations |
| `src/xflats/scraper/boligsiden.py` | Add logging, docstrings, fix 1 broad except |
| `src/xflats/extraction/gemini.py` | Add docstrings, fix bare dict types |
| `src/xflats/storage/chromadb.py` | Add logging, docstrings, type `collection` params, fix bare dicts |
| `src/xflats/notifications/telegram.py` | Add logging, docstrings, type `collection` param, fix bare dict |

### Implementation Details

1. **pyproject.toml** — Enable ruff rules:
   - `"D"` (pydocstyle) with `convention = "google"`
   - `"T20"` (flake8-print — catches print() calls)
   - `"BLE"` (flake8-blind-except)
   - `"ANN"` (flake8-annotations) — consider, may be noisy
   - Keep line-length = 88

2. **Logging setup** — Add `logging.getLogger(__name__)` to each module, replace all `print()` with `logger.info/warning/error`. Add basic config in `main.py`.

3. **Docstrings** — Google style for all public functions:
   ```python
   def foo(bar: str) -> int:
       """Short description.

       Args:
           bar: Description of bar.

       Returns:
           Description of return value.

       Raises:
           ValueError: If bar is empty.
       """
   ```

4. **Type annotations** — Full annotations for all functions. Use `chromadb.Collection` for collection params, `dict[str, Any]` instead of bare `dict`.

5. **Exception handling** — Replace broad `except Exception` with specific types where possible (e.g., `requests.RequestException`, `json.JSONDecodeError`). Where truly generic needed, add comment explaining why.

## Verification

- [ ] `ruff check src/` passes with new rules
- [ ] `python -m py_compile` on all source files
- [ ] `just test` passes (no behavioral changes)
- [ ] `just lint` passes

## Risks & Open Questions

| Risk | Mitigation |
|------|-----------|
| ANN rules may be too noisy (e.g., `self` annotations) | Ignore `ANN101`/`ANN102` |
| Some broad excepts are intentional (top-level crash prevention) | Add `noqa` with comment explaining why |
| Logging config may affect Docker cron output | Use `logging.basicConfig` with StreamHandler (same as print to stdout) |

## Issues Log

| # | Issue | Resolution |
|---|-------|------------|
| | | |
