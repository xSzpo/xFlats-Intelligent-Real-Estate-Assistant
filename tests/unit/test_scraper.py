"""Tests for xflats.scraper.boligsiden and related utils."""

from unittest.mock import MagicMock, patch

from xflats.scraper.boligsiden import (
    _preprocess_html,
    check_crawl_permission,
    extract_adresse_urls,
)
from xflats.utils import remove_url_parameters


class TestExtractAdresseUrls:
    def test_extract_adresse_urls(self, sample_html):
        urls = extract_adresse_urls(sample_html)
        url_strs = [str(u) for u in urls]
        # Should find 3 /adresse/ links (query params stripped)
        assert len(urls) == 3
        assert any("vesterbrogade" in u for u in url_strs)
        assert any("noerrebrogade" in u for u in url_strs)
        assert any("amagerbrogade" in u for u in url_strs)
        # Should NOT include /about
        assert not any("about" in u for u in url_strs)
        # Query params should be stripped
        assert not any("?" in u for u in url_strs)

    def test_extract_adresse_urls_no_links(self):
        html = "<html><body><p>No links here</p></body></html>"
        urls = extract_adresse_urls(html)
        assert urls == []


class TestPreprocessHtml:
    def test_preprocess_html(self):
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <nav>Navigation</nav>
            <script>var x = 1;</script>
            <style>.foo { color: red; }</style>
            <div>  Hello   World  </div>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        result = _preprocess_html(html)
        # Noise tags stripped
        assert "Navigation" not in result
        assert "var x" not in result
        assert "color: red" not in result
        assert "Footer" not in result
        # Content preserved, whitespace collapsed
        assert "Hello World" in result
        assert "  " not in result


class TestCheckCrawlPermission:
    @patch("xflats.scraper.boligsiden.urlopen")
    def test_check_crawl_permission_allowed(self, mock_urlopen):
        robots_txt = b"User-agent: *\nAllow: /\n"
        mock_response = MagicMock()
        mock_response.read.return_value = robots_txt
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        assert check_crawl_permission("https://example.com/adresse/test") is True

    @patch("xflats.scraper.boligsiden.urlopen")
    def test_check_crawl_permission_denied(self, mock_urlopen):
        robots_txt = b"User-agent: *\nDisallow: /adresse/\n"
        mock_response = MagicMock()
        mock_response.read.return_value = robots_txt
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        assert check_crawl_permission("https://example.com/adresse/test") is False


class TestRemoveUrlParameters:
    def test_remove_url_parameters(self):
        assert (
            remove_url_parameters("/adresse/test?ref=search&page=1") == "/adresse/test"
        )
        assert remove_url_parameters("/adresse/test") == "/adresse/test"
        assert (
            remove_url_parameters("https://example.com/path?q=1")
            == "https://example.com/path"
        )
