"""
Real Estate Scraper for Warsaw Property Listings

This module scrapes property listings from otodom.pl, processes them using
Google's Gemini AI, stores them in ChromaDB, and sends notifications via Telegram.
"""

import datetime
import hashlib
import os
import time
from typing import Any

import chromadb
import requests
from bs4 import BeautifulSoup
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.api_core import retry
from google.genai import types
from pydantic import BaseModel
from utils import (
    add_offers_to_db,
    check_crawl_permission,
    chromadb_check_if_document_exists,
    create_offer_text,
    fetch_and_preprocess,
    fetch_html,
    fix_json,
    geocode_address,
    get_price_point,
    get_secret,
    is_retriable,
    remove_url_parameters,
)

# Configuration constants
OFFER_VERSION = 2
MAX_RETRIES = 2  # Reduced to minimize wasted API calls on failures
DEFAULT_PAGES_TO_OPEN = 3
DEFAULT_NUMBER_OF_ROOMS = 2
GET_OFFERS_FROM_X_LAST_MIN = (
    35  # Check for offers from last 35 min (covers cron cycle + processing time)
)
CHROMADB_DEFAULT_PORT = 8000
DB_NAME = "real-estate-offers-warsaw"
LISTINGS_PER_API_CALL = (
    15  # Process 15 listings per API call (gemini-2.5-flash has high limits)
)
MAX_URLS_PER_PAGE = 15  # Limit URLs to prevent excessive API usage
API_DELAY_SECONDS = 6  # Delay between API calls to respect 15 RPM limit (60s/15 = 4s, use 6s for safety)

# API endpoints and templates
BASE_URL = (
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/wiele-lokalizacji?limit=36"
    "&ownerTypeSingleSelect=ALL&locations="
    "%5Bmazowieckie%2Fwarszawa%2Fwarszawa%2Fwarszawa%2F"
    "srodmiescie%2Fpowisle%2Cmazowieckie%2Fwarszawa%2Fwarszawa%2Fwarszawa%2Fsrodmiescie"
    "%2Fsrodmiescie-poludniowe%5D&by=LATEST&direction=DESC&mapBounds="
    "21.066349201274893%2C52.25105396894465%2C20.97234973673827%2C52.208541518407955&page={page}"
)
URLS = [
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EhxNb2tvdG93c2thLCBXYXJzemF3YSwgUG9sYW5kIi4qLAoUChIJda0NfeXMHkcR7L6Sd1NEGagSFAoSCQGfhppmzB5HEfzT6ogqvvBy&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EiFwbGFjIFpiYXdpY2llbGEsIFdhcnN6YXdhLCBQb2xza2EiLiosChQKEgmtkxz95cweRxFGXhIa9lSLfhIUChIJAZ-GmmbMHkcR_NPqiCq-8HI&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EiVwbGFjIFRyemVjaCBLcnp5xbx5LCBXYXJzemF3YSwgUG9sc2thIi4qLAoUChIJv_4Ov_DMHkcRm2NwCF96cMgSFAoSCQGfhppmzB5HEfzT6ogqvvBy&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EhdSYWRuYSwgV2Fyc3phd2EsIFBvbHNrYSIuKiwKFAoSCde_MB1czB5HEWAAdCtVCRPuEhQKEgkBn4aaZsweRxH80-qIKr7wcg&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EhdEb2JyYSwgV2Fyc3phd2EsIFBvbGFuZCIuKiwKFAoSCWG6LwlczB5HESljV7Y7kcsCEhQKEgkBn4aaZsweRxH80-qIKr7wcg&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=ChIJ5Wpy21vMHkcRcoBshj3qoT8&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=Eh5MZXN6Y3p5xYRza2EsIFdhcnN6YXdhLCBQb2xhbmQiLiosChQKEglfHbCsXsweRxF3C_D5GnoF8RIUChIJAZ-GmmbMHkcR_NPqiCq-8HI&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
    "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/cala-polska?distanceRadius=200&placeId=EhdTb2xlYywgV2Fyc3phd2EsIFBvbHNrYSIuKiwKFAoSCZWIUub-zB5HEdrtcc9N_qQsEhQKEgkBn4aaZsweRxH80-qIKr7wcg&limit=48&ownerTypeSingleSelect=ALL&by=LATEST&direction=DESC",
]


PROMPT_TEMPLATE = """
You are an expert in extracting apartment listings from cleaned HTML text. Your task is to extract key structured information and present it in **valid JSON format**.

Please follow these instructions **precisely**:

1. **Translate all text to English**, except for the **Address**, which must remain in its original Polish language.
2. **Description**: Keep it concise (1-2 sentences). Only include information NOT already captured in other fields (address, price, size, rooms, year, floor). Focus on unique features like: furnishing status, elevator availability, specific amenities, or notable positives/negatives. Avoid repeating structured data.
3. **Address**: Extract in this format: `Street Name Number, PostalCode City, Country` (keep in Polish)
   - Do NOT include unit/floor/apartment numbers in the address
   - If only district/neighborhood is available, use: `District, City, Country`
4. **Price**: Extract as an integer in PLN, no commas or currency signs (e.g., `950000`). If missing, use `null`.
5. **Rent** (Czynsz): Extract monthly rent/maintenance fee as integer in PLN (e.g., `800`). If missing, use `null`.
6. **Area (m2)**: Extract as an integer or float (e.g., `58` or `58.5`). If missing, use `null`.
7. **Number of Rooms**: Extract total number of rooms as an integer. If missing, use `null`.
8. **Year Built**: Extract the year the building was constructed (e.g., `2015`). If missing, use `null`.
9. **Energy Label**: Extract as a single uppercase letter (`A`, `B`, etc.). If not available, use `null`.
10. **Balcony**: Return `true` if a balcony or terrace is mentioned; otherwise, `false`.
11. **Floor**: Extract floor information in format "current/total" (e.g., `"2/5"` for 2nd floor of 5-story building) or just floor number (e.g., `"2"`, `"ground"`). If missing, use `null`.
12. **URL**: Extract the full link to the listing.

Ensure the output is **JSON only**, with no explanation or additional text.

Cleaned HTMLs:

{html_content}

JSON output:
"""

SOURCE_TEMPLATE = """
---------------------
Offer #{i}
URL: {url}  
SOURCE:
{text}
"""

EXAMPLE_TEXT = """
```json
[
    {
        "address": "Powiśle, Warszawa, Polska",
        "description": "Furnished, move-in ready. West-facing balcony with courtyard view. No elevator.",
        "floor": "5/7",
        "price": 950000,
        "rent": 850,
        "area_m2": 58,
        "number_of_rooms": 3,
        "year_built": 2015,
        "energy_label": "B",
        "balcony": true,
        "url": "https://www.otodom.pl/pl/oferta/example-listing-12345"
    }
]
"""


class Offers(BaseModel):
    """Pydantic model for property offer data."""

    address: str
    description: str
    floor: str | None
    price: int | None
    rent: int | None
    area_m2: float | None
    number_of_rooms: int | None
    year_built: int | None
    energy_label: str | None
    balcony: bool | None
    url: str


class ListOfOffers(BaseModel):
    """Container for multiple property offers."""

    offers: list[Offers]


class Config:
    """Configuration management for the scraper."""

    def __init__(self):
        self.profile_name = os.getenv("AWS_PROFILE", None)
        self.chromadb_ip = os.getenv("CHROMADB_IP", "chromadb")
        self.number_of_pages_to_open = int(
            os.getenv("NUMBER_OF_PAGES_TO_OPEN", DEFAULT_PAGES_TO_OPEN)
        )
        self.number_of_rooms = int(
            os.getenv("NUMBER_OF_ROOMS", DEFAULT_NUMBER_OF_ROOMS)
        )

        # Initialize API credentials
        self.telegram_token = get_secret(
            secret_id="telegram-011337673661",
            key="TOKEN",
            profile_name=self.profile_name,
        )
        self.telegram_chat_id = get_secret(
            secret_id="telegram-011337673661",
            key="CHAT_ID",
            profile_name=self.profile_name,
        )
        self.genai_api_key = get_secret(
            secret_id="gemini-011337673661",
            key="GOOGLE_API_KEY",
            profile_name=self.profile_name,
        )


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Google's Gemini API."""

    def __init__(self, client: genai.Client, *args, **kwargs):
        self.client = client
        self.document_mode = True
        super().__init__(*args, **kwargs)

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def extract_otodom_urls(html_content: str) -> list[str]:
    """
    Extract property listing URLs from otodom.pl search results page.

    Args:
        html_content: HTML content of the search results page

    Returns:
        List of property listing URLs
    """
    soup = BeautifulSoup(html_content, "html.parser")
    urls = []

    # Otodom.pl uses data-cy="listing-item-link" or similar patterns
    # Try multiple selectors to find listing links
    link_selectors = [
        'a[data-cy="listing-item-link"]',
        'a[href*="/pl/oferta/"]',
        "a.css-rvjxyq",  # Common class for listing links
    ]

    for selector in link_selectors:
        links = soup.select(selector)
        for link in links:
            href = link.get("href", "")
            if href and "/pl/oferta/" in href:
                # Make absolute URL if relative
                if href.startswith("/"):
                    href = f"https://www.otodom.pl{href}"
                elif not href.startswith("http"):
                    href = f"https://www.otodom.pl/{href}"

                # Clean URL (remove query parameters except essential ones)
                if "?" in href:
                    href = href.split("?")[0]

                if href not in urls:
                    urls.append(href)

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


class RealEstateScraper:
    """Main scraper class for Warsaw real estate listings."""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.genai_api_key)
        self.collection = self._setup_vector_database()

    def _setup_vector_database(self) -> chromadb.Collection:
        """Initialize ChromaDB collection with Gemini embeddings."""
        print("Initializing vector database...")

        embed_fn = GeminiEmbeddingFunction(self.client)
        embed_fn.document_mode = True

        if self.config.chromadb_ip:
            chroma_client = chromadb.HttpClient(
                host=self.config.chromadb_ip, port=CHROMADB_DEFAULT_PORT
            )
        else:
            chroma_client = chromadb.PersistentClient()

        collection = chroma_client.get_or_create_collection(
            name=DB_NAME, embedding_function=embed_fn
        )
        print("Vector database initialized")
        return collection

    def _generate_url_hash(self, url: str | Any) -> str:
        """Generate a unique hash for a given URL."""
        url_str = str(url)
        return hashlib.shake_128(url_str.encode()).hexdigest(8)

    def _extract_new_urls(self, url: str) -> list[str]:
        """Extract new URLs from a given search URL that don't exist in the database."""
        print(f"Processing URL: {url}")

        if not check_crawl_permission(url):
            raise ValueError(f"Crawling not permitted: {url}")

        html_content = fetch_html(url)
        urls = extract_otodom_urls(html_content)

        new_urls = []
        for url in urls:
            url_str = str(url)
            url_hash = self._generate_url_hash(url_str)
            if not chromadb_check_if_document_exists(url_hash, self.collection):
                new_urls.append(url_str)

        return new_urls

    def _process_offers_with_ai(
        self, offers_source: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Process offer sources using Gemini AI to extract structured data."""
        if not offers_source:
            return []

        source_content = ""
        for i, offer in enumerate(offers_source, 1):
            source_content += SOURCE_TEMPLATE.format(i=i, **offer)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ListOfOffers,
                    max_output_tokens=8192,
                ),
                contents=[
                    PROMPT_TEMPLATE.format(html_content=source_content),
                    EXAMPLE_TEXT,
                ],
            )
        except Exception as e:
            error_str = str(e)
            print(f"ERROR: Gemini API call failed: {e}")

            # If quota exceeded, check if we should wait and retry
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay if available (usually in format "retry in Xs")
                import re

                retry_match = re.search(r"retry in ([0-9.]+)s", error_str)
                if retry_match:
                    retry_seconds = float(retry_match.group(1))
                    print(
                        f"Quota exceeded. API suggests waiting {retry_seconds:.0f}s. Skipping this batch."
                    )
                else:
                    print(
                        "Daily quota exceeded (20 requests/day). Wait until quota resets."
                    )
            return []

        result = fix_json(response.text)

        if result and isinstance(result, dict) and "offers" in result:
            return result["offers"]
        elif result and isinstance(result, list):
            return result

        print("WARNING: No valid offers extracted from API response!")
        return []

    def _enrich_offer_data(self, offer: dict[str, Any]) -> None:
        """Enrich offer data with geocoding and transport information."""
        # Clean URL and generate ID
        if offer.get("url"):
            url_str = str(offer["url"])
            offer["url"] = remove_url_parameters(url_str)
            offer["id"] = self._generate_url_hash(offer["url"])

        offer["version"] = OFFER_VERSION
        offer["create_date"] = datetime.datetime.today().timestamp()

        # Add location data
        try:
            address = offer.get("address")
            if address:
                # For Warsaw addresses, ensure "Polska" or "Poland" is included
                if "Polska" not in address and "Poland" not in address:
                    address = f"{address}, Polska"

                # Strip 'ul.' prefix which causes geocoding failures
                geocode_address_clean = address.replace("ul. ", "").replace("ul.", "")

                lat, lon = geocode_address(geocode_address_clean)
                offer["lat"], offer["long"] = lat, lon
                print(f"Retrieved geocode for {address}")
        except Exception as e:
            print(f"Failed to retrieve geocode data: {e}")

    def _process_url_with_retry(self, url: str) -> list[dict[str, Any]]:
        """Process a single search URL with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                new_urls = self._extract_new_urls(url)

                if not new_urls:
                    print(f"No new URLs found for: {url}")
                    return []

                print(f"Processing {len(new_urls)} listings from search URL")

                # Fetch and preprocess all URLs
                offers_source = []
                for url in new_urls:
                    text = fetch_and_preprocess(url, mode="two_requests")
                    if text:
                        offers_source.append({"url": url, "text": text})

                if not offers_source:
                    print("No valid content found")
                    return []

                print(f"Fetched content for {len(offers_source)} listings")

                # Process with AI in batches to avoid overwhelming the model
                all_offers = []
                for batch_start in range(0, len(offers_source), LISTINGS_PER_API_CALL):
                    batch = offers_source[
                        batch_start : batch_start + LISTINGS_PER_API_CALL
                    ]
                    batch_num = (batch_start // LISTINGS_PER_API_CALL) + 1
                    total_batches = (
                        len(offers_source) - 1
                    ) // LISTINGS_PER_API_CALL + 1

                    print(
                        f"Processing batch {batch_num}/{total_batches} ({len(batch)} listings)..."
                    )
                    batch_offers = self._process_offers_with_ai(batch)

                    if batch_offers:
                        all_offers.extend(batch_offers)
                        print(
                            f"Extracted {len(batch_offers)} offers from batch {batch_num}"
                        )

                    # Delay between batches to respect RPM limits (15 RPM = 1 request per 4s minimum)
                    if batch_num < total_batches:
                        time.sleep(API_DELAY_SECONDS)

                if all_offers:
                    print(
                        f"Retrieved total of {len(all_offers)} offers from search URL"
                    )

                    # Enrich each offer with additional data
                    for offer in all_offers:
                        self._enrich_offer_data(offer)
                        time.sleep(1)  # Rate limiting for geocoding API

                return all_offers

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing URL, attempt {attempt}/{MAX_RETRIES}: {e}")

                # If it's a quota/rate limit error, wait longer
                if (
                    "429" in error_msg
                    or "RESOURCE_EXHAUSTED" in error_msg
                    or "quota" in error_msg.lower()
                ):
                    wait_time = 60 * attempt  # 60s, 120s, 180s
                    print(
                        f"Rate limit hit. Waiting {wait_time} seconds before retry..."
                    )
                    time.sleep(wait_time)
                elif attempt < MAX_RETRIES:
                    time.sleep(attempt * 2)  # Exponential backoff for other errors
                else:
                    print(f"Failed to process URL after {MAX_RETRIES} attempts")
                    return []

        return []

    def _deduplicate_offers(self, offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate offers based on their ID."""
        seen_ids = set()
        unique_offers = []

        for offer in offers:
            offer_id = offer.get("id")
            if offer_id and offer_id not in seen_ids:
                seen_ids.add(offer_id)
                unique_offers.append(offer)

        return unique_offers

    def scrape_historical_listings(self) -> list[dict[str, Any]]:
        """Scrape historical listings from multiple search URLs."""
        print("Starting data collection process for Warsaw apartments")
        all_results = []

        for i, url in enumerate(URLS, 1):
            print(f"\n=== Processing search URL {i}/{len(URLS)} ===")
            url_results = self._process_url_with_retry(url)
            all_results.extend(url_results)
            time.sleep(2)  # Be respectful to the server

        print(f"Collected {len(all_results)} total listings")

        # Remove duplicates
        unique_results = self._deduplicate_offers(all_results)
        print(f"Found {len(unique_results)} unique listings")

        return unique_results

    def get_recent_offers(self) -> list[dict[str, Any]]:
        """Retrieve recent offers that match criteria."""
        now = datetime.datetime.now()
        cutoff_time = (
            now - datetime.timedelta(minutes=GET_OFFERS_FROM_X_LAST_MIN)
        ).timestamp()

        results = self.collection.get(
            include=["metadatas"],
            where={
                "$and": [
                    {"create_date": {"$gt": cutoff_time}},
                    {"number_of_rooms": {"$gte": self.config.number_of_rooms}},
                ]
            },
        )

        return results.get("metadatas", [])

    def send_telegram_notifications(self, offers: list[dict[str, Any]]) -> None:
        """Send offer notifications via Telegram."""
        if not offers:
            print("No apartment listings to share on Telegram")
            return

        print(f"Sending {len(offers)} listings to Telegram")

        # Enrich with price points and sort
        for offer in offers:
            offer["price_point"] = get_price_point(offer, self.collection)

        offers.sort(key=lambda x: x.get("price_point", 0))

        # Send notifications
        for offer in offers:
            offer_text = create_offer_text(offer)
            print(f"Sending: {offer_text}")

            telegram_url = (
                f"https://api.telegram.org/bot{self.config.telegram_token}/"
                f"sendMessage?chat_id={self.config.telegram_chat_id}&text={offer_text}"
            )

            try:
                response = requests.get(telegram_url)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Failed to send Telegram message: {e}")

    def run(self) -> None:
        """Main execution method."""
        # Scrape and store historical listings
        historical_offers = self.scrape_historical_listings()

        if historical_offers:
            print(
                f"Adding {len(historical_offers)} historical listings to vector database"
            )
            add_offers_to_db(self.collection, historical_offers)

            # Filter newly scraped offers by room criteria before sending notifications
            filtered_offers = [
                offer
                for offer in historical_offers
                if offer.get("number_of_rooms", 0) >= self.config.number_of_rooms
            ]

            # Send notifications only for newly scraped offers (prevents duplicates)
            self.send_telegram_notifications(filtered_offers)
        else:
            print("No new offers to send")


def main() -> None:
    """Main entry point."""
    try:
        config = Config()
        scraper = RealEstateScraper(config)
        scraper.run()
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
