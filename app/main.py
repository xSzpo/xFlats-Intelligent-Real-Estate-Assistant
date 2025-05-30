"""
Real Estate Scraper for Danish Property Listings

This module scrapes property listings from boligsiden.dk, processes them using
Google's Gemini AI, stores them in ChromaDB, and sends notifications via Telegram.
"""

import datetime
import hashlib
import os
import time
from typing import Any

import chromadb
import requests
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
    extract_adresse_urls,
    fetch_and_preprocess,
    fetch_html,
    fix_json,
    geocode_address,
    get_price_point,
    get_public_transport_stations,
    get_secret,
    is_retriable,
    offer_to_text,
    remove_url_parameters,
)

# Configuration constants
OFFER_VERSION = 2
MAX_RETRIES = 3
DEFAULT_PAGES_TO_OPEN = 3
DEFAULT_NUMBER_OF_ROOMS = 2
GET_OFFERS_FROM_X_LAST_MIN = 5
CHROMADB_DEFAULT_PORT = 8000
DB_NAME = "real-estate-offers"

# API endpoints and templates
BASE_URL = (
    "https://www.boligsiden.dk/tilsalg/villa,ejerlejlighed?sortAscending=true"
    "&mapBounds=7.780294,54.501948,15.330305,57.896401&priceMax=7000000"
    "&polygon=12.555001,55.714439|12.544964,55.711152|12.535566,55.708713|12.523383,55.700403|"
    "12.513564,55.690885|12.507604,55.674192|12.508089,55.656840|12.521769,55.648585|"
    "12.534702,55.642731|12.564876,55.614388|12.591917,55.614270|12.599055,55.649692|"
    "12.605518,55.649361|12.615303,55.649093|12.628699,55.649335|12.641590,55.649906|"
    "12.636977,55.665739|12.626008,55.676732|12.636641,55.686489|12.654036,55.720127|"
    "12.602392,55.730897|12.555001,55.714439&page={page}"
)

PROMPT_TEMPLATE = """
You are an expert in extracting apartment listings from cleaned HTML text. Your task is to extract key structured information and present it in **valid JSON format**.

Please follow these instructions **precisely**:

1. **Translate all text to English**, except for the **Address**, which must remain in its original language.
2. **Description**: in English, craft a neutral, informative overview covering:
  - Flat layout and standout positives/negatives  
  - Natural light (e.g. “bright, east-facing”)  
  - Condition (e.g. “newly built”, “recently renovated”, “well-maintained older building”)  
  - View (e.g. “courtyard”, “street-facing with greenery”)  
  - Neighborhood vibe (e.g. “quiet residential”, “central and well-connected”) 
3. **Address**: Extract in this format: `Street Name Number, PostalCode City, Country`  
   - Do NOT include unit/floor/apartment numbers in the address
4. **Price**: Extract as an integer, no commas or currency signs (e.g., `3250000`). If missing, use `null`.
5. **Area (m2)**: Extract as an integer (e.g., `87`). If missing, use `null`.
6. **Number of Rooms**: Extract total number of rooms as an integer. If missing, use `null`.
7. **Year Built**: Extract the year the building was constructed (e.g., `2006`). If missing, use `null`.
8. **Energy Label**: Extract as a single uppercase letter (`A`, `B`, etc.). If not available, use `null`.
9. **Balcony**: Return `true` if a balcony or terrace is mentioned; otherwise, `false`.
10. **URL**: Extract the full link to the listing.

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
        "address": "Engholmene, 2450 København SV, Denmark",
        "description": "Apartment boasts abundant natural light and a spacious west-facing balcony overlooking the canal and marina. The contemporary interior is move-in ready, featuring high-quality materials. The neighborhood offers plenty of greenery, cafés, promenades, and convenient metro access.",
        "floor": "5",
        "price": 6195000,
        "area_m2": 91,
        "number_of_rooms": 2,
        "year_built": 2019,
        "energy_label": "A",
        "balcony": "true",
        "url": "https://www.boligsiden.dk/adresse/engholmene-2450-koebenhavn-sv-eksempel"
    }
]
"""


class Offers(BaseModel):
    """Pydantic model for property offer data."""

    address: str
    description: str
    floor: str
    price: int
    area_m2: int
    number_of_rooms: int
    year_built: int
    energy_label: str
    balcony: str
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
            secret_id="telegram-274181059559",
            key="TOKEN",
            profile_name=self.profile_name,
        )
        self.telegram_chat_id = get_secret(
            secret_id="telegram-274181059559",
            key="CHAT_ID",
            profile_name=self.profile_name,
        )
        self.genai_api_key = get_secret(
            secret_id="gemini-274181059559",
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
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


class RealEstateScraper:
    """Main scraper class for Danish real estate listings."""

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
        # Convert to string in case it's a Pydantic HttpUrl object
        url_str = str(url)
        return hashlib.shake_128(url_str.encode()).hexdigest(8)

    def _extract_new_urls(self, page: int) -> list[str]:
        """Extract new URLs from a given page that don't exist in the database."""
        page_url = BASE_URL.format(page=page)
        print(f"Processing page {page}: {page_url}")

        if not check_crawl_permission(page_url):
            raise ValueError(f"Crawling not permitted: {page_url}")

        html_content = fetch_html(page_url)
        urls = extract_adresse_urls(html_content)

        new_urls = []
        for url in urls:
            # Convert to string in case it's a Pydantic HttpUrl object
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

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
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

        return fix_json(response.text) or []

    def _enrich_offer_data(self, offer: dict[str, Any]) -> None:
        """Enrich offer data with geocoding and transport information."""
        # Clean URL and generate ID
        if offer.get("url"):
            # Convert to string in case it's a Pydantic HttpUrl object
            url_str = str(offer["url"])
            offer["url"] = remove_url_parameters(url_str)
            offer["id"] = self._generate_url_hash(offer["url"])

        offer["version"] = OFFER_VERSION
        offer["create_date"] = datetime.datetime.today().timestamp()

        # Add location data
        try:
            address = offer.get("address")
            if address:
                lat, lon = geocode_address(address)
                offer["lat"], offer["long"] = lat, lon

                # Fetch public transport stations
                stations = get_public_transport_stations(lat=lat, lon=lon)
                offer.update(stations)
                print(f"Retrieved location data for {address}")
        except Exception as e:
            print(f"Failed to retrieve geocode data: {e}")

    def _process_page_with_retry(self, page: int) -> list[dict[str, Any]]:
        """Process a single page with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                new_urls = self._extract_new_urls(page)

                if not new_urls:
                    print(f"No new URLs found on page {page}")
                    return []

                print(f"Found {len(new_urls)} new URLs on page {page}")

                # Fetch and preprocess content
                offers_source = []
                for url in new_urls:
                    text = fetch_and_preprocess(url, mode="two_requests")
                    if text:
                        offers_source.append({"url": url, "text": text})

                # Process with AI
                offers = self._process_offers_with_ai(offers_source)

                if offers:
                    print(f"Retrieved {len(offers)} offers from page {page}")

                    # Enrich each offer with additional data
                    for offer in offers:
                        self._enrich_offer_data(offer)
                        time.sleep(1)  # Rate limiting

                return offers

            except Exception as e:
                print(
                    f"Error processing page {page}, attempt {attempt}/{MAX_RETRIES}: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(attempt)  # Exponential backoff
                else:
                    print(f"Failed to process page {page} after {MAX_RETRIES} attempts")
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
        """Scrape historical listings from multiple pages."""
        print("Starting data collection process")
        all_results = []

        for page in range(1, self.config.number_of_pages_to_open + 1):
            page_results = self._process_page_with_retry(page)
            all_results.extend(page_results)

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
                    {"subways": {"$eq": True}},
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
            for offer in historical_offers:
                print(offer_to_text(offer))
            add_offers_to_db(self.collection, historical_offers)

        # Get and send recent offers
        recent_offers = self.get_recent_offers()
        self.send_telegram_notifications(recent_offers)


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
