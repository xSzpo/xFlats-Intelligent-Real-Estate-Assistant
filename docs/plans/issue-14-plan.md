# Implementation Plan — Issue #14

## Metadata
- **Issue**: #14 — feat: WAW region support (Warsaw scraping into main)
- **Branch**: `feat/14/waw-region-support`
- **Date**: 2026-05-16

## Status

| Step | Status | Notes |
|------|--------|-------|
| Context gathered | done | EC2 WAW branch analyzed via git diff |
| Plan approved | pending | |
| Implementation | pending | |
| Tested | pending | |
| PR created | pending | |

## Problem Statement

EC2 runs diverged `region/waw` branch with old `app/` layout. WAW scraping works in production but isn't merged into `main` (new `src/xflats/` structure). Need to port Warsaw/otodom.pl scraping into main codebase with multi-region config so both WAW and CPH can coexist.

## Non-Goals

- Copenhagen/Denmark activation (separate issue)
- EC2 deployment / decommissioning `region/waw` branch
- Terraform changes
- Docker multi-region cron (future)

## Current State

Main has CPH-only hardcoded setup:
- `main.py`: hardcoded boligsiden.dk BASE_URL with CPH polygon
- `extraction/gemini.py`: CPH examples in prompt, gemini-2.0-flash, text-embedding-004
- `storage/chromadb.py`: single collection `real-estate-offers`
- `notifications/telegram.py`: DKK currency, subway filter
- `config/secrets.py`: single set of Telegram/Gemini secrets

WAW branch (`region/waw`) changes:
- Target site: otodom.pl (8 neighborhood URLs vs 1 paginated URL)
- URL extraction: `extract_otodom_urls()` (new function)
- Pydantic model: added `rent`/`floor` fields, nullable fields
- AI model: gemini-2.5-flash, gemini-embedding-001
- Secrets: different AWS secret IDs
- DB collection: `real-estate-offers-warsaw`
- Prompt: Warsaw-specific examples (PLN, Polish addresses)
- Batching: 15 listings/call, 6s delay, 429 handling
- Notifications: no subway filter, 35min window
- Utils: browser headers, geocoding fixes, None metadata filtering

## Proposed Changes

### Architecture Decision

Introduce a **region config system**. Each region is a Python dataclass/dict defining all region-specific parameters. The main entry point selects region via `REGION` env var.

### Files to Modify

| File | Change |
|------|--------|
| `src/xflats/config/secrets.py` | Add `REGION` env var, per-region secret IDs, region config dataclass |
| `src/xflats/config/regions.py` | **NEW** — Region configs (WAW, CPH) with URLs, polygon, currency, site, collection name, prompt examples, model versions, thresholds |
| `src/xflats/main.py` | Accept region config, remove hardcoded BASE_URL, dispatch to correct scraper based on site |
| `src/xflats/scraper/__init__.py` | Scraper registry/factory |
| `src/xflats/scraper/boligsiden.py` | Parameterize with region config (keep as CPH scraper) |
| `src/xflats/scraper/otodom.py` | **NEW** — Port WAW scraper from region/waw branch, adapted to new structure |
| `src/xflats/extraction/gemini.py` | Parameterize prompt, model name, embedding model from region config. Add `rent`/`floor` to Offers model (nullable). Batch support. |
| `src/xflats/storage/chromadb.py` | Parameterize collection name from region config. Filter None metadata. |
| `src/xflats/notifications/telegram.py` | Parameterize currency, subway filter, time window from region config |
| `src/xflats/utils.py` | Add browser headers, geocoding country suffix, None metadata filter, timeout fixes |

### Implementation Details

#### 1. Region Config (`config/regions.py`)

```python
@dataclass
class RegionConfig:
    name: str                    # "waw" | "cph"
    site: str                    # "otodom" | "boligsiden"
    urls: list[str]              # search URLs
    collection_name: str         # ChromaDB collection
    currency: str                # "PLN" | "DKK"
    country: str                 # "Polska" | "Denmark"
    gemini_model: str            # "gemini-2.5-flash" | "gemini-2.0-flash"
    embedding_model: str         # "gemini-embedding-001" | "text-embedding-004"
    prompt_examples: dict        # region-specific AI prompt examples
    telegram_secret_id: str      # AWS secret ID for Telegram
    gemini_secret_id: str        # AWS secret ID for Gemini
    batch_size: int              # listings per AI call
    batch_delay_s: float         # delay between batches
    notify_window_min: int       # GET_OFFERS_FROM_X_LAST_MIN
    notify_filter_subway: bool   # whether to filter by subway
    min_rooms: int               # minimum rooms filter
```

#### 2. Otodom Scraper (`scraper/otodom.py`)

Port `extract_otodom_urls()` from WAW branch. Adapt to new module structure. Key differences from boligsiden:
- Multiple URLs (neighborhoods) vs single paginated URL
- Different HTML structure for URL extraction
- Different robots.txt rules

#### 3. Gemini Changes

- Make `Offers` model fields nullable (add `rent: int | None`, `floor: str | None`)
- Parameterize prompt template with region examples
- Parameterize model names
- Add batch processing support (multiple listings per call)
- Add 429/RESOURCE_EXHAUSTED retry logic

#### 4. Storage Changes

- Collection name from region config
- Filter `None` values from metadata before ChromaDB insert

#### 5. Notification Changes

- Currency from region config
- Subway filter conditional on region
- Time window from region config

#### 6. Utils Changes

- Browser User-Agent headers on all requests
- Geocoding: append country name
- Explicit timeouts on all requests
- Reduced Overpass retries/backoff

#### 7. Entry Point

```python
# main.py
region = os.environ.get("REGION", "waw")  # default WAW since that's production
config = get_region_config(region)
scraper = RealEstateScraper(config, region_config)
scraper.run()
```

## Verification

- [ ] Python syntax check passes on all new/modified files
- [ ] `uv run python -c "from xflats.config.regions import get_region_config; print(get_region_config('waw'))"` works
- [ ] `uv run python -c "from xflats.scraper.otodom import extract_otodom_urls; print('OK')"` imports clean
- [ ] `just lint` passes
- [ ] `just test` passes (existing tests)
- [ ] Docker build succeeds

## Risks & Open Questions

1. **AI model version**: WAW branch uses gemini-2.5-flash. Should we upgrade CPH too or keep separate? → Keep separate per region config.
2. **Embedding model**: WAW uses gemini-embedding-001 vs CPH text-embedding-004. Mixing in same DB would cause issues → separate collections solves this.
3. **Rate limiting**: WAW branch has sophisticated 429 handling. Should be in shared code, not region-specific.
4. **Secrets**: WAW uses different AWS secret IDs. Need both sets in Secrets Manager.
5. **Default region**: Setting WAW as default since it's the active production region.
6. **Nullable fields**: Adding `rent`/`floor` to Offers model — CPH won't populate these, they'll be None. OK.

## Issues Log

| # | Issue | Resolution |
|---|-------|------------|
| | | |
