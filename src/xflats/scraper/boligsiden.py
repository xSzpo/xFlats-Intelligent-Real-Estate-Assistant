"""Scraping logic for boligsiden.dk listings."""

import logging
import re
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup, Comment
from pydantic import HttpUrl, ValidationError

logger = logging.getLogger(__name__)


def extract_adresse_urls(
    html_content: str, base_url: str = "https://www.boligsiden.dk"
) -> list[HttpUrl]:
    """Parse HTML and return unique validated HttpUrl for ``/adresse/`` links.

    Args:
        html_content: Raw HTML string to parse.
        base_url: Base URL used to resolve relative paths.

    Returns:
        List of validated ``HttpUrl`` objects for unique adresse links.
    """
    from xflats.utils import remove_url_parameters

    soup = BeautifulSoup(html_content, "html.parser")
    pattern = re.compile(r"^/adresse/")
    raw_paths = {tag["href"] for tag in soup.find_all("a", href=pattern)}

    validated_urls = set()
    for path in raw_paths:
        clean_path = remove_url_parameters(str(path))
        full = urljoin(base_url, clean_path)
        try:
            validated_urls.add(HttpUrl(full))
        except ValidationError:
            continue
    return list(validated_urls)


def check_crawl_permission(target_page: str) -> bool:
    """Check whether crawling is allowed for a URL per its robots.txt.

    Parses the ``robots.txt`` for the wildcard user-agent and evaluates
    allow/disallow rules using longest-match semantics.

    Args:
        target_page: Fully-qualified URL to check.

    Returns:
        ``True`` if crawling is permitted (or robots.txt is unreachable),
        ``False`` otherwise.
    """
    parsed = urlparse(target_page)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or "/"
    robots_url = urljoin(base_url, "/robots.txt")

    try:
        with urlopen(robots_url, timeout=5) as response:
            robots_content = response.read().decode("utf-8", errors="ignore")
    except OSError as e:
        logger.warning("Error fetching %s: %s", robots_url, e)
        return True

    rules: list[tuple[str, str]] = []
    current_agent = None

    for line in robots_content.splitlines():
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        field, value = [part.strip() for part in line.split(":", 1)]
        field = field.lower()
        if field == "user-agent":
            current_agent = value
        elif current_agent == "*" and field in ("allow", "disallow"):
            rules.append((field, value))

    best_match: str | None = None
    best_length = -1
    for directive, pattern in rules:
        if not pattern:
            continue
        regex = "^" + re.escape(pattern).replace("\\*", ".*")
        if re.search(regex, path):
            if len(pattern) > best_length:
                best_match = directive
                best_length = len(pattern)
            elif len(pattern) == best_length and directive == "allow":
                best_match = directive

    return best_match != "disallow"


def fetch_html(url: str) -> str:
    """Fetch a page and return its raw HTML.

    Args:
        url: URL to fetch.

    Returns:
        Raw HTML content as a string.

    Raises:
        requests.HTTPError: If the response status code indicates an error.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def _preprocess_html(html: str) -> str:
    """Clean raw HTML for downstream text extraction.

    Strips ``<head>``, comments, noise tags (nav, footer, scripts, etc.),
    preserves JSON-LD blocks, removes all attributes, and collapses whitespace.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned plain-text string with JSON-LD blocks prepended.
    """
    soup = BeautifulSoup(html, "html.parser")
    if soup.head:
        soup.head.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    jsonld = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        if tag.string:
            jsonld.append(tag.string.strip())
        tag.decompose()

    for name in (
        "script",
        "style",
        "meta",
        "link",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "input",
        "button",
        "select",
        "option",
        "textarea",
        "canvas",
        "iframe",
        "noscript",
    ):
        for t in soup.find_all(name):
            t.decompose()

    for t in soup.find_all(True):
        t.attrs = {}

    text = (soup.body or soup).get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    for block in jsonld:
        text = f"<JSON-LD>{block}</JSON-LD> " + text
    return text


def fetch_and_preprocess(
    url: str, timeout: float = 5.0, max_bytes: int = 10_000, mode: str = "two_requests"
) -> str | None:
    """Fetch and preprocess a page, returning cleaned text.

    Supports two modes: ``two_requests`` (peek first, then full fetch) and
    ``single_request`` (stream everything in one pass). Returns ``None`` for
    404-like responses or network errors.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        max_bytes: Max bytes to read in the peek request (``two_requests`` mode).
        mode: Either ``'two_requests'`` or ``'single_request'``.

    Returns:
        Cleaned text string, or ``None`` if page is not found or unreachable.

    Raises:
        ValueError: If *mode* is not a recognised value.
    """
    not_found_markers = [
        '<html id="__next_error__">',
        "NEXT_NOT_FOUND",
        "Siden findes ikke!",
    ]

    try:
        if mode == "two_requests":
            resp = requests.get(url, timeout=timeout, stream=True)
            if 400 <= resp.status_code < 500:
                return None
            total = 0
            buf: list[str] = []
            for chunk in resp.iter_content(1024, decode_unicode=True):
                buf.append(str(chunk))
                total += len(chunk)
                if total >= max_bytes:
                    break
            snippet = "".join(buf)
            resp.close()
            if any(marker in snippet for marker in not_found_markers):
                return None
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            html = resp.text

        elif mode == "single_request":
            resp = requests.get(url, timeout=timeout, stream=True)
            if 400 <= resp.status_code < 500:
                return None
            buf2: list[str] = []
            for chunk in resp.iter_content(1024, decode_unicode=True):
                buf2.append(str(chunk))
            resp.close()
            html = "".join(buf2)
            if any(marker in html for marker in not_found_markers):
                return None
        else:
            raise ValueError("mode must be 'two_requests' or 'single_request'")

    except requests.RequestException:
        return None

    return _preprocess_html(html)
