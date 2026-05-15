"""
Real Estate Scraper for Danish Property Listings

Entry point — scrapes boligsiden.dk, extracts via Gemini AI,
stores in ChromaDB, sends Telegram notifications.
"""

import datetime
import hashlib
import time
from typing import Any

from google import genai

from xflats.config.secrets import Config
from xflats.extraction.gemini import GeminiEmbeddingFunction, process_offers_with_ai
from xflats.notifications.telegram import send_telegram_notifications
from xflats.scraper.boligsiden import (
    check_crawl_permission,
    extract_adresse_urls,
    fetch_and_preprocess,
    fetch_html,
)
from xflats.storage.chromadb import (
    add_offers_to_db,
    check_if_document_exists,
    get_recent_offers,
    setup_vector_database,
)
from xflats.utils import (
    geocode_address,
    get_public_transport_stations,
    remove_url_parameters,
)

# Configuration constants
OFFER_VERSION = 2
MAX_RETRIES = 3
GET_OFFERS_FROM_X_LAST_MIN = 5

# API endpoints
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


class RealEstateScraper:
    """Main scraper class for Danish real estate listings."""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.genai_api_key)
        self.collection = setup_vector_database(
            config.chromadb_ip, GeminiEmbeddingFunction(self.client)
        )

    def _generate_url_hash(self, url: str) -> str:
        url_str = str(url)
        return hashlib.shake_128(url_str.encode()).hexdigest(8)

    def _extract_new_urls(self, page: int) -> list[str]:
        page_url = BASE_URL.format(page=page)
        print(f"Processing page {page}: {page_url}")

        if not check_crawl_permission(page_url):
            raise ValueError(f"Crawling not permitted: {page_url}")

        html_content = fetch_html(page_url)
        urls = extract_adresse_urls(html_content)

        new_urls = []
        for url in urls:
            url_str = str(url)
            url_hash = self._generate_url_hash(url_str)
            if not check_if_document_exists(url_hash, self.collection):
                new_urls.append(url_str)

        return new_urls

    def _enrich_offer_data(self, offer: dict[str, Any]) -> None:
        if offer.get("url"):
            url_str = str(offer["url"])
            offer["url"] = remove_url_parameters(url_str)
            offer["id"] = self._generate_url_hash(offer["url"])

        offer["version"] = OFFER_VERSION
        offer["create_date"] = datetime.datetime.now(datetime.timezone.utc).timestamp()

        try:
            address = offer.get("address")
            if address:
                lat, lon = geocode_address(address)
                offer["lat"], offer["long"] = lat, lon
                stations = get_public_transport_stations(lat=lat, lon=lon)
                offer.update(stations)
                print(f"Retrieved location data for {address}")
        except Exception as e:
            print(f"Failed to retrieve geocode data: {e}")

    def _process_page_with_retry(self, page: int) -> list[dict[str, Any]]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                new_urls = self._extract_new_urls(page)
                if not new_urls:
                    print(f"No new URLs found on page {page}")
                    return []

                print(f"Found {len(new_urls)} new URLs on page {page}")

                offers_source = []
                for url in new_urls:
                    text = fetch_and_preprocess(url, mode="two_requests")
                    if text:
                        offers_source.append({"url": url, "text": text})

                offers = process_offers_with_ai(self.client, offers_source)

                if offers:
                    print(f"Retrieved {len(offers)} offers from page {page}")
                    for offer in offers:
                        self._enrich_offer_data(offer)
                        time.sleep(1)

                return offers

            except Exception as e:
                print(
                    f"Error processing page {page}, attempt {attempt}/{MAX_RETRIES}: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(attempt)
                else:
                    print(f"Failed to process page {page} after {MAX_RETRIES} attempts")
                    return []

        return []

    def _deduplicate_offers(self, offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_ids = set()
        unique_offers = []
        for offer in offers:
            offer_id = offer.get("id")
            if offer_id and offer_id not in seen_ids:
                seen_ids.add(offer_id)
                unique_offers.append(offer)
        return unique_offers

    def scrape_historical_listings(self) -> list[dict[str, Any]]:
        print("Starting data collection process")
        all_results = []
        for page in range(1, self.config.number_of_pages_to_open + 1):
            page_results = self._process_page_with_retry(page)
            all_results.extend(page_results)

        print(f"Collected {len(all_results)} total listings")
        unique_results = self._deduplicate_offers(all_results)
        print(f"Found {len(unique_results)} unique listings")
        return unique_results

    def run(self) -> None:
        historical_offers = self.scrape_historical_listings()

        if historical_offers:
            print(
                f"Adding {len(historical_offers)} historical listings to vector database"
            )
            add_offers_to_db(self.collection, historical_offers)

        now = datetime.datetime.now()
        cutoff_time = (
            now - datetime.timedelta(minutes=GET_OFFERS_FROM_X_LAST_MIN)
        ).timestamp()
        recent_offers = get_recent_offers(
            self.collection, cutoff_time, self.config.number_of_rooms
        )
        send_telegram_notifications(
            recent_offers,
            self.collection,
            self.config.telegram_token,
            self.config.telegram_chat_id,
        )


def main() -> None:
    try:
        config = Config()
        scraper = RealEstateScraper(config)
        scraper.run()
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
