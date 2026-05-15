"""Region configuration for multi-region real estate scraping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionConfig:
    """Configuration for a specific geographic region.

    Each region defines its target real estate site, search URLs,
    AI model versions, secret IDs, and notification preferences.

    Attributes:
        name: Short region identifier (e.g. "waw", "cph").
        display_name: Human-readable region name.
        site: Target real estate site identifier ("otodom" or "boligsiden").
        urls: List of search URLs to scrape.
        collection_name: ChromaDB collection name for this region.
        currency: Currency code for price display (e.g. "PLN", "DKK").
        country: Country name appended to addresses for geocoding.
        gemini_model: Gemini model name for structured extraction.
        embedding_model: Gemini embedding model name.
        prompt_address_example: Example address for AI prompt.
        prompt_description_example: Example description for AI prompt.
        prompt_url_example: Example URL for AI prompt.
        prompt_price_example: Example price integer for AI prompt.
        prompt_rent_example: Example rent integer for AI prompt (None if N/A).
        telegram_secret_id: AWS Secrets Manager ID for Telegram credentials.
        gemini_secret_id: AWS Secrets Manager ID for Gemini API key.
        batch_size: Number of listings per AI API call.
        batch_delay_s: Delay in seconds between AI API batches.
        notify_window_min: Minutes to look back for recent offers.
        notify_filter_subway: Whether to filter notifications by subway access.
        min_rooms: Default minimum rooms for notification filter.
        max_retries: Maximum retry attempts for page processing.
        pages_to_open: Default number of pages to scrape (for paginated sites).
        offer_version: Version tag for stored offers.
        fetch_mode: HTTP fetch mode ("two_requests" or "single_request").
        use_browser_headers: Whether to send browser-like HTTP headers.
        address_cleanup: Whether to strip address prefixes (e.g. "ul.").
    """

    name: str
    display_name: str
    site: str
    urls: list[str]
    collection_name: str
    currency: str
    country: str
    gemini_model: str
    embedding_model: str
    prompt_address_example: str
    prompt_description_example: str
    prompt_url_example: str
    prompt_price_example: int
    prompt_rent_example: int | None
    telegram_secret_id: str
    gemini_secret_id: str
    batch_size: int = 15
    batch_delay_s: float = 6.0
    notify_window_min: int = 35
    notify_filter_subway: bool = False
    min_rooms: int = 2
    max_retries: int = 2
    pages_to_open: int = 3
    offer_version: int = 2
    fetch_mode: str = "two_requests"
    use_browser_headers: bool = True
    address_cleanup: bool = False


# Browser-like headers for sites that block simple requests.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


WAW_URLS = [
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EhxNb2tvdG93c2thLCBXYXJzemF3YSwgUG9sYW5kIi4qLAoUChIJ"
        "da0NfeXMHkcR7L6Sd1NEGagSFAoSCQGfhppmzB5HEfzT6ogqvvBy"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EiFwbGFjIFpiYXdpY2llbGEsIFdhcnN6YXdhLCBQb2xza2EiLiosChQK"
        "Egmtkxz95cweRxFGXhIa9lSLfhIUChIJAZ-GmmbMHkcR_NPqiCq-8HI"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EiVwbGFjIFRyemVjaCBLcnp5xbx5LCBXYXJzemF3YSwgUG9sc2thIi4qLAoU"
        "ChIJv_4Ov_DMHkcRm2NwCF96cMgSFAoSCQGfhppmzB5HEfzT6ogqvvBy"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EhdSYWRuYSwgV2Fyc3phd2EsIFBvbHNrYSIuKiwKFAoSCde_MB1czB5HEWAAd"
        "CtVCRPuEhQKEgkBn4aaZsweRxH80-qIKr7wcg"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EhdEb2JyYSwgV2Fyc3phd2EsIFBvbGFuZCIuKiwKFAoSCWG6LwlczB5HESJR"
        "V7Y7kcsCEhQKEgkBn4aaZsweRxH80-qIKr7wcg"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=ChIJ5Wpy21vMHkcRcoBshj3qoT8"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=Eh5MZXN6Y3p5xYRza2EsIFdhcnN6YXdhLCBQb2xhbmQiLiosChQKEglf"
        "HbCsXsweRxF3C_D5GnoF8RIUChIJAZ-GmmbMHkcR_NPqiCq-8HI"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
    (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?"
        "distanceRadius=200&placeId=EhdTb2xlYywgV2Fyc3phd2EsIFBvbHNrYSIuKiwKFAoSCZWIUub-zB5HEdrt"
        "cc9N_qQsEhQKEgkBn4aaZsweRxH80-qIKr7wcg"
        "&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC"
    ),
]


WAW_CONFIG = RegionConfig(
    name="waw",
    display_name="Warsaw",
    site="otodom",
    urls=WAW_URLS,
    collection_name="real-estate-offers-warsaw",
    currency="PLN",
    country="Polska",
    gemini_model="gemini-2.5-flash",
    embedding_model="gemini-embedding-001",
    prompt_address_example="Powiśle, Warszawa, Polska",
    prompt_description_example=(
        "Furnished, move-in ready. West-facing balcony "
        "with courtyard view. No elevator."
    ),
    prompt_url_example="https://www.otodom.pl/pl/oferta/example-listing-12345",
    prompt_price_example=950000,
    prompt_rent_example=850,
    telegram_secret_id="telegram-011337673661",
    gemini_secret_id="gemini-011337673661",
    batch_size=15,
    batch_delay_s=6.0,
    notify_window_min=35,
    notify_filter_subway=False,
    min_rooms=2,
    max_retries=2,
    pages_to_open=3,
    offer_version=2,
    fetch_mode="two_requests",
    use_browser_headers=True,
    address_cleanup=True,
)

# CPH placeholder — will be activated in a separate issue.
CPH_URLS = [
    (
        "https://www.boligsiden.dk/tilsalg/villa,ejerlejlighed?sortAscending=true"
        "&mapBounds=7.780294,54.501948,15.330305,57.896401&priceMax=7000000"
        "&polygon=12.555001,55.714439|12.544964,55.711152|12.535566,55.708713|"
        "12.523383,55.700403|12.513564,55.690885|12.507604,55.674192|"
        "12.508089,55.656840|12.521769,55.648585|12.534702,55.642731|"
        "12.564876,55.614388|12.591917,55.614270|12.599055,55.649692|"
        "12.605518,55.649361|12.615303,55.649093|12.628699,55.649335|"
        "12.641590,55.649906|12.636977,55.665739|12.626008,55.676732|"
        "12.636641,55.686489|12.654036,55.720127|12.602392,55.730897|"
        "12.555001,55.714439&page={{page}}"
    ),
]

CPH_CONFIG = RegionConfig(
    name="cph",
    display_name="Copenhagen",
    site="boligsiden",
    urls=CPH_URLS,
    collection_name="real-estate-offers",
    currency="DKK",
    country="Denmark",
    gemini_model="gemini-2.0-flash",
    embedding_model="text-embedding-004",
    prompt_address_example="Engholmene, 2450 København SV, Denmark",
    prompt_description_example=(
        "Apartment boasts abundant natural light and a spacious "
        "west-facing balcony overlooking the canal and marina."
    ),
    prompt_url_example=(
        "https://www.boligsiden.dk/adresse/engholmene-2450-koebenhavn-sv-eksempel"
    ),
    prompt_price_example=6195000,
    prompt_rent_example=None,
    telegram_secret_id="telegram-274181059559",
    gemini_secret_id="gemini-274181059559",
    batch_size=1,
    batch_delay_s=1.0,
    notify_window_min=5,
    notify_filter_subway=True,
    min_rooms=2,
    max_retries=3,
    pages_to_open=3,
    offer_version=2,
    fetch_mode="two_requests",
    use_browser_headers=False,
    address_cleanup=False,
)

_REGIONS: dict[str, RegionConfig] = {
    "waw": WAW_CONFIG,
    "cph": CPH_CONFIG,
}


def get_region_config(region: str) -> RegionConfig:
    """Look up region configuration by name.

    Args:
        region: Region identifier (e.g. "waw", "cph").

    Returns:
        The ``RegionConfig`` for the requested region.

    Raises:
        ValueError: If the region name is not recognised.
    """
    region = region.lower()
    if region not in _REGIONS:
        available = ", ".join(sorted(_REGIONS))
        raise ValueError(f"Unknown region {region!r}. Available: {available}")
    return _REGIONS[region]
