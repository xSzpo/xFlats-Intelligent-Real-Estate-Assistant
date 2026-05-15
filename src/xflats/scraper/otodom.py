"""Scraping logic for otodom.pl property listings (Warsaw)."""

import logging
import time

import requests
from bs4 import BeautifulSoup

from xflats.config.regions import BROWSER_HEADERS

logger = logging.getLogger(__name__)

BASE_URL = "https://www.otodom.pl"


def extract_otodom_urls(html_content: str) -> list[str]:
    """Parse otodom.pl search results HTML and extract listing URLs.

    Tries multiple CSS selectors to find listing links, since otodom.pl
    may change class names across deployments.

    Args:
        html_content: Raw HTML string from an otodom.pl search page.

    Returns:
        Deduplicated list of absolute listing URLs.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    link_selectors = [
        'a[data-cy="listing-item-link"]',
        'a[href*="/pl/oferta/"]',
        "a.css-rvjxyq",
    ]

    seen: set[str] = set()
    urls: list[str] = []

    for selector in link_selectors:
        matches = soup.select(selector)
        if not matches and "css-" in selector:
            logger.warning(
                "CSS selector %r matched zero elements — class name may have changed",
                selector,
            )
        for link in matches:
            href = str(link.get("href", ""))
            if not href or "/pl/oferta/" not in href:
                continue

            if href.startswith("/"):
                href = f"{BASE_URL}{href}"
            elif not href.startswith("http"):
                href = f"{BASE_URL}/{href}"

            # Strip query parameters.
            if "?" in href:
                href = href.split("?")[0]

            if href not in seen:
                seen.add(href)
                urls.append(href)

    return urls


def fetch_html(url: str, use_browser_headers: bool = True) -> str:
    """Fetch a page and return its raw HTML.

    Sends browser-like headers by default to avoid being blocked
    by otodom.pl.

    Args:
        url: URL to fetch.
        use_browser_headers: Whether to include browser-like request
            headers.

    Returns:
        Raw HTML content as a string.

    Raises:
        requests.HTTPError: If the response status code indicates an error.
    """
    headers = BROWSER_HEADERS if use_browser_headers else {}
    time.sleep(1)  # Respectful delay.
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text
