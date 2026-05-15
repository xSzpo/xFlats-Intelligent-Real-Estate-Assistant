"""Real Estate Scraper — multi-region entry point.

Scrapes property listings from region-specific sites (otodom.pl, boligsiden.dk),
extracts structured data via Gemini AI, stores in ChromaDB, and sends
Telegram notifications. Region is selected via the ``REGION`` env var.
"""

import datetime
import hashlib
import logging
import time
from typing import Any

import requests
from google import genai

from xflats.config.regions import RegionConfig
from xflats.config.secrets import Config
from xflats.extraction.gemini import (
    PROMPT_TEMPLATE,
    GeminiEmbeddingFunction,
    process_offers_with_ai,
)
from xflats.notifications.telegram import send_telegram_notifications
from xflats.scraper.boligsiden import (
    check_crawl_permission,
    extract_adresse_urls,
    fetch_and_preprocess,
)
from xflats.scraper.boligsiden import (
    fetch_html as boligsiden_fetch_html,
)
from xflats.scraper.otodom import (
    extract_otodom_urls,
)
from xflats.scraper.otodom import (
    fetch_html as otodom_fetch_html,
)
from xflats.storage.chromadb import (
    add_offers_to_db,
    check_if_document_exists,
    setup_vector_database,
)
from xflats.utils import (
    geocode_address,
    get_public_transport_stations,
    remove_url_parameters,
)

logger = logging.getLogger(__name__)


def _build_prompt_template(region: RegionConfig) -> str:
    """Build a region-specific AI prompt template.

    Customises the base prompt with region-appropriate instructions
    for address language, currency, rent field, and floor format.

    Args:
        region: Region configuration to derive prompt details from.

    Returns:
        A prompt template string with ``{html_content}`` placeholder.
    """
    region_instruction = ""
    if region.country == "Polska":
        region_instruction = " (keep in Polish)"

    rent_instruction = ""
    if region.prompt_rent_example is not None:
        rent_instruction = (
            "5. **Rent** (Czynsz): Extract monthly rent/maintenance fee "
            f"as integer in {region.currency} (e.g., "
            f"`{region.prompt_rent_example}`). If missing, use `null`.\n"
        )

    return PROMPT_TEMPLATE.format_map(
        {
            "html_content": "{html_content}",
            "region_instruction": region_instruction,
            "rent_instruction": rent_instruction,
        }
    )


def _build_example_text(region: RegionConfig) -> str:
    """Build region-specific example JSON for the AI prompt.

    Args:
        region: Region configuration with example values.

    Returns:
        A JSON code-block string with one example offer.
    """
    rent_line = ""
    if region.prompt_rent_example is not None:
        rent_line = f'\n        "rent": {region.prompt_rent_example},'

    return f'''
```json
[
    {{
        "address": "{region.prompt_address_example}",
        "description": "{region.prompt_description_example}",
        "floor": "5/7",
        "price": {region.prompt_price_example},{rent_line}
        "area_m2": 58,
        "number_of_rooms": 3,
        "year_built": 2015,
        "energy_label": "B",
        "balcony": true,
        "url": "{region.prompt_url_example}"
    }}
]
```
'''


class RealEstateScraper:
    """Main scraper class supporting multiple regions."""

    def __init__(self, config: Config) -> None:
        """Initialize scraper with configuration.

        Args:
            config: Application configuration containing API keys and settings.
        """
        self.config = config
        self.region = config.region_config
        self.client = genai.Client(api_key=config.genai_api_key)

        # Resolve embedding model name — some models need "models/" prefix.
        embed_model = self.region.embedding_model
        if not embed_model.startswith("models/"):
            embed_model = f"models/{embed_model}"

        self.collection = setup_vector_database(
            config.chromadb_ip,
            GeminiEmbeddingFunction(self.client, model=embed_model),
            collection_name=self.region.collection_name,
        )

        # Build region-specific prompt.
        self.prompt_template = _build_prompt_template(self.region)
        self.example_text = _build_example_text(self.region)

    # ------------------------------------------------------------------
    # URL hashing
    # ------------------------------------------------------------------

    def _generate_url_hash(self, url: str) -> str:
        """Generate a short hash for a URL.

        Args:
            url: The URL to hash.

        Returns:
            A 16-character hex digest of the URL.
        """
        return hashlib.shake_128(str(url).encode()).hexdigest(8)

    # ------------------------------------------------------------------
    # URL extraction — dispatches per site
    # ------------------------------------------------------------------

    def _extract_new_urls(self, search_url: str) -> list[str]:
        """Extract listing URLs not yet in the database.

        Dispatches to the correct site-specific URL extractor based on
        the region's ``site`` setting.

        Args:
            search_url: The search/listing page URL to scrape.

        Returns:
            A list of new listing URLs not already stored.

        Raises:
            ValueError: If crawling is not permitted or site is unknown.
        """
        logger.info("Processing URL: %s", search_url)

        if not check_crawl_permission(search_url):
            raise ValueError(f"Crawling not permitted: {search_url}")

        if self.region.site == "otodom":
            html_content = otodom_fetch_html(
                search_url, use_browser_headers=self.region.use_browser_headers
            )
            urls = extract_otodom_urls(html_content)
        elif self.region.site == "boligsiden":
            html_content = boligsiden_fetch_html(search_url)
            urls = [str(u) for u in extract_adresse_urls(html_content)]
        else:
            raise ValueError(f"Unknown site: {self.region.site!r}")

        new_urls: list[str] = []
        for url in urls:
            url_str = str(url)
            url_hash = self._generate_url_hash(url_str)
            if not check_if_document_exists(url_hash, self.collection):
                new_urls.append(url_str)

        return new_urls

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich_offer_data(self, offer: dict[str, Any]) -> None:
        """Enrich an offer dict with computed fields and geolocation data.

        Args:
            offer: Mutable offer dictionary to enrich in-place.
        """
        if offer.get("url"):
            url_str = str(offer["url"])
            offer["url"] = remove_url_parameters(url_str)
            offer["id"] = self._generate_url_hash(offer["url"])

        offer["version"] = self.region.offer_version
        offer["create_date"] = datetime.datetime.now(datetime.timezone.utc).timestamp()

        try:
            address = offer.get("address")
            if address:
                lat, lon = geocode_address(
                    address,
                    country=self.region.country,
                    address_cleanup=self.region.address_cleanup,
                )
                offer["lat"], offer["long"] = lat, lon

                if self.region.notify_filter_subway:
                    stations = get_public_transport_stations(lat=lat, lon=lon)
                    offer.update(stations)

                logger.info("Retrieved location data for %s", address)
        except (ValueError, requests.RequestException) as e:
            logger.error("Failed to retrieve geocode data: %s", e)

    # ------------------------------------------------------------------
    # AI processing
    # ------------------------------------------------------------------

    def _process_url_with_retry(self, search_url: str) -> list[dict[str, Any]]:
        """Process a search URL with retry and batching logic.

        Args:
            search_url: The search page URL to process.

        Returns:
            A list of extracted offer dictionaries, empty on failure.
        """
        for attempt in range(1, self.region.max_retries + 1):
            try:
                new_urls = self._extract_new_urls(search_url)
                if not new_urls:
                    logger.info("No new URLs found")
                    return []

                logger.info("Found %d new URLs", len(new_urls))

                # Fetch and preprocess content.
                offers_source: list[dict[str, str]] = []
                for url in new_urls:
                    text = fetch_and_preprocess(url, mode=self.region.fetch_mode)
                    if text:
                        offers_source.append({"url": url, "text": text})

                if not offers_source:
                    logger.info("No valid content fetched")
                    return []

                logger.info("Fetched content for %d listings", len(offers_source))

                # Process in batches.
                all_offers: list[dict[str, Any]] = []
                batch_size = self.region.batch_size
                total_batches = (len(offers_source) - 1) // batch_size + 1

                for batch_start in range(0, len(offers_source), batch_size):
                    batch = offers_source[batch_start : batch_start + batch_size]
                    batch_num = batch_start // batch_size + 1

                    logger.info(
                        "Processing batch %d/%d (%d listings)",
                        batch_num,
                        total_batches,
                        len(batch),
                    )

                    batch_offers = process_offers_with_ai(
                        self.client,
                        batch,
                        model=self.region.gemini_model,
                        prompt_template=self.prompt_template,
                        example_text=self.example_text,
                    )

                    if batch_offers:
                        all_offers.extend(batch_offers)
                        logger.info(
                            "Extracted %d offers from batch %d",
                            len(batch_offers),
                            batch_num,
                        )

                    if batch_num < total_batches:
                        time.sleep(self.region.batch_delay_s)

                # Enrich offers.
                for offer in all_offers:
                    self._enrich_offer_data(offer)
                    time.sleep(1)

                return all_offers

            except requests.RequestException as e:
                error_msg = str(e)
                logger.error(
                    "Error attempt %d/%d: %s",
                    attempt,
                    self.region.max_retries,
                    e,
                )

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = 60 * attempt
                    logger.warning("Rate limit hit. Waiting %ds...", wait_time)
                    time.sleep(wait_time)
                elif attempt < self.region.max_retries:
                    time.sleep(attempt * 2)
                else:
                    logger.error("Failed after %d attempts", self.region.max_retries)
                    return []

        return []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_offers(self, offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate offers based on their ID.

        Args:
            offers: List of offer dictionaries, possibly with duplicates.

        Returns:
            A deduplicated list preserving first-seen order.
        """
        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        for offer in offers:
            offer_id = offer.get("id")
            if offer_id and offer_id not in seen_ids:
                seen_ids.add(offer_id)
                unique.append(offer)
        return unique

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def scrape_listings(self) -> list[dict[str, Any]]:
        """Scrape listings from all configured search URLs.

        For paginated sites (boligsiden), iterates over page numbers.
        For multi-URL sites (otodom), iterates over the URL list.

        Returns:
            A deduplicated list of offer dictionaries.
        """
        logger.info("Starting %s data collection", self.region.display_name)
        all_results: list[dict[str, Any]] = []

        if self.region.site == "boligsiden":
            for page in range(1, self.config.number_of_pages_to_open + 1):
                url = self.region.urls[0].format(page=page)
                results = self._process_url_with_retry(url)
                all_results.extend(results)
        else:
            for i, url in enumerate(self.region.urls, 1):
                logger.info("=== Search URL %d/%d ===", i, len(self.region.urls))
                results = self._process_url_with_retry(url)
                all_results.extend(results)
                time.sleep(2)

        logger.info("Collected %d total listings", len(all_results))
        unique = self._deduplicate_offers(all_results)
        logger.info("Found %d unique listings", len(unique))
        return unique

    def run(self) -> None:
        """Execute the full scraping, storage, and notification pipeline."""
        offers = self.scrape_listings()

        if offers:
            logger.info("Adding %d listings to vector database", len(offers))
            add_offers_to_db(self.collection, offers)

            # Send notifications for newly scraped offers.
            filtered = [
                o
                for o in offers
                if (o.get("number_of_rooms") or 0) >= self.config.number_of_rooms
            ]
            send_telegram_notifications(
                filtered,
                self.collection,
                self.config.telegram_token,
                self.config.telegram_chat_id,
                currency=self.region.currency,
            )
        else:
            logger.info("No new offers found")


def main() -> None:
    """Run the real estate scraper application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        config = Config()
        logger.info(
            "Starting scraper for region: %s (%s)",
            config.region_config.display_name,
            config.region,
        )
        scraper = RealEstateScraper(config)
        scraper.run()
    except Exception as e:  # noqa: BLE001
        logger.error("Fatal error in main execution: %s", e)
        raise


if __name__ == "__main__":
    main()
