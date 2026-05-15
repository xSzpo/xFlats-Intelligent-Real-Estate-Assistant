"""Tests for xflats.scraper.otodom module."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from xflats.scraper.otodom import extract_otodom_urls, fetch_html

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def otodom_search_html():
    """Sample otodom.pl search results page with listing links."""
    return """
    <html><body>
        <a data-cy="listing-item-link"
           href="/pl/oferta/mieszkanie-mokotow-123">Listing 1</a>
        <a data-cy="listing-item-link"
           href="/pl/oferta/mieszkanie-ursynow-456?tracking=abc">Listing 2</a>
        <a href="/pl/oferta/mieszkanie-wola-789" class="css-rvjxyq">Listing 3</a>
        <a href="/about">About</a>
        <a href="/pl/oferta/mieszkanie-mokotow-123">Duplicate</a>
    </body></html>
    """


# ---------------------------------------------------------------------------
# extract_otodom_urls
# ---------------------------------------------------------------------------


class TestExtractOtodomUrls:
    """Tests for extract_otodom_urls."""

    def test_extracts_listing_urls(self, otodom_search_html):
        """Should extract all unique /pl/oferta/ URLs."""
        urls = extract_otodom_urls(otodom_search_html)

        assert len(urls) == 3
        assert "https://www.otodom.pl/pl/oferta/mieszkanie-mokotow-123" in urls
        assert "https://www.otodom.pl/pl/oferta/mieszkanie-ursynow-456" in urls
        assert "https://www.otodom.pl/pl/oferta/mieszkanie-wola-789" in urls

    def test_strips_query_params(self, otodom_search_html):
        """Query parameters should be removed from URLs."""
        urls = extract_otodom_urls(otodom_search_html)

        assert not any("?" in u for u in urls)

    def test_deduplicates(self, otodom_search_html):
        """Duplicate hrefs should appear only once."""
        urls = extract_otodom_urls(otodom_search_html)

        mokotow = [u for u in urls if "mokotow" in u]
        assert len(mokotow) == 1

    def test_excludes_non_listing_links(self, otodom_search_html):
        """Links without /pl/oferta/ in href should be excluded."""
        urls = extract_otodom_urls(otodom_search_html)

        assert not any("about" in u for u in urls)

    def test_empty_html(self):
        """Empty HTML body returns no URLs."""
        urls = extract_otodom_urls("<html><body></body></html>")

        assert urls == []

    def test_no_listing_links(self):
        """HTML with links but none matching /pl/oferta/ returns empty."""
        html = '<html><body><a href="/contact">Contact</a></body></html>'
        urls = extract_otodom_urls(html)

        assert urls == []

    def test_malformed_html(self):
        """Malformed HTML should not raise, returns best-effort results."""
        html = '<a href="/pl/oferta/test-123" <broken tag>'
        urls = extract_otodom_urls(html)

        # BeautifulSoup handles malformed HTML gracefully
        assert isinstance(urls, list)

    def test_absolute_urls_unchanged(self):
        """Already-absolute URLs should not get double-prefixed."""
        html = '<a href="https://www.otodom.pl/pl/oferta/abs-1">Link</a>'
        urls = extract_otodom_urls(html)

        assert urls == ["https://www.otodom.pl/pl/oferta/abs-1"]

    def test_relative_url_without_slash(self):
        """Relative href without leading slash is skipped (no /pl/oferta/ match)."""
        html = '<a href="pl/oferta/rel-1">Link</a>'
        urls = extract_otodom_urls(html)

        # Source checks for "/pl/oferta/" (with leading slash) so bare
        # relative paths without "/" prefix are filtered out.
        assert urls == []


# ---------------------------------------------------------------------------
# fetch_html
# ---------------------------------------------------------------------------


class TestFetchHtml:
    """Tests for fetch_html."""

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_success(self, mock_get, mock_sleep):
        """Successful fetch returns response text."""
        mock_response = MagicMock()
        mock_response.text = "<html>OK</html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_html("https://www.otodom.pl/pl/oferta/test")

        assert result == "<html>OK</html>"
        mock_get.assert_called_once()
        mock_sleep.assert_called_once_with(1)

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_sends_browser_headers_by_default(self, mock_get, mock_sleep):
        """Default call should include BROWSER_HEADERS."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetch_html("https://example.com")

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"] != {}

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_no_browser_headers(self, mock_get, mock_sleep):
        """use_browser_headers=False sends empty headers."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetch_html("https://example.com", use_browser_headers=False)

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"] == {}

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_http_error_raised(self, mock_get, mock_sleep):
        """HTTP errors should propagate as HTTPError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403")
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            fetch_html("https://example.com")

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_timeout(self, mock_get, mock_sleep):
        """Connection timeout should propagate."""
        mock_get.side_effect = requests.ConnectionError("Timeout")

        with pytest.raises(requests.ConnectionError):
            fetch_html("https://example.com")

    @patch("xflats.scraper.otodom.time.sleep")
    @patch("xflats.scraper.otodom.requests.get")
    def test_timeout_parameter(self, mock_get, mock_sleep):
        """requests.get called with timeout=30."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetch_html("https://example.com")

        assert mock_get.call_args.kwargs["timeout"] == 30
