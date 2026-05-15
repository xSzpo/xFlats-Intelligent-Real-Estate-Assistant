"""Tests for xflats.notifications.telegram."""

from unittest.mock import MagicMock, patch

from xflats.notifications.telegram import create_offer_text, send_telegram_notifications


class TestCreateOfferText:
    def test_create_offer_text(self, sample_offer):
        sample_offer["price_point"] = 1.05
        text = create_offer_text(sample_offer)
        assert "Vesterbrogade 42" in text
        assert "3,250,000 DKK" in text
        assert "Rooms: 3" in text
        assert "87 m2" in text
        assert "105.00%" in text

    def test_create_offer_text_with_subway(self, sample_offer):
        sample_offer["price_point"] = 0.95
        text = create_offer_text(sample_offer)
        assert "Enghave Plads" in text
        assert "Vesterport" in text

    def test_create_offer_text_no_subway(self, sample_offer):
        sample_offer["subways"] = False
        sample_offer["price_point"] = 1.0
        text = create_offer_text(sample_offer)
        assert "Subway(s):" in text


class TestSendTelegramNotifications:
    @patch("xflats.notifications.telegram.requests.post")
    @patch("xflats.notifications.telegram.get_price_point")
    def test_send_telegram_notifications(self, mock_price_point, mock_post, sample_offer):
        mock_price_point.return_value = 1.05
        mock_post.return_value = MagicMock(status_code=200)

        collection = MagicMock()
        send_telegram_notifications(
            offers=[sample_offer],
            collection=collection,
            telegram_token="fake-token",
            telegram_chat_id="12345",
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "api.telegram.org" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"]["chat_id"] == "12345"
        mock_post.return_value.raise_for_status.assert_called_once()

    @patch("xflats.notifications.telegram.requests.post")
    @patch("xflats.notifications.telegram.get_price_point")
    def test_send_telegram_notifications_empty(self, mock_price_point, mock_post):
        collection = MagicMock()
        send_telegram_notifications(
            offers=[],
            collection=collection,
            telegram_token="fake-token",
            telegram_chat_id="12345",
        )
        mock_post.assert_not_called()
