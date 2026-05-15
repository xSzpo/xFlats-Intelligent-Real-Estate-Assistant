"""Tests for xflats.storage.chromadb."""

from xflats.storage.chromadb import (
    add_offers_to_db,
    check_if_document_exists,
    get_recent_offers,
    offer_to_text,
)


class TestCheckIfDocumentExists:
    def test_check_if_document_exists_true(self, mock_chromadb_collection):
        mock_chromadb_collection.get.return_value = {"ids": ["existing-id"]}
        assert check_if_document_exists("existing-id", mock_chromadb_collection) is True
        mock_chromadb_collection.get.assert_called_once_with(ids=["existing-id"])

    def test_check_if_document_exists_false(self, mock_chromadb_collection):
        mock_chromadb_collection.get.return_value = {"ids": []}
        assert check_if_document_exists("missing-id", mock_chromadb_collection) is False


class TestOfferToText:
    def test_offer_to_text(self, sample_offer):
        text = offer_to_text(sample_offer)
        assert "Address: Vesterbrogade 42" in text
        assert "Area: 87 m²" in text
        assert "Rooms: 3" in text
        assert "Balcony: True" in text
        assert "Year built: 1905" in text
        assert "Energy label: C" in text


class TestAddOffersToDb:
    def test_add_offers_to_db(self, mock_chromadb_collection, sample_offer):
        offers = [sample_offer]
        add_offers_to_db(mock_chromadb_collection, offers)
        mock_chromadb_collection.add.assert_called_once()
        call_kwargs = mock_chromadb_collection.add.call_args
        assert call_kwargs.kwargs["ids"] == ["test-offer-001"]
        assert len(call_kwargs.kwargs["documents"]) == 1
        assert call_kwargs.kwargs["metadatas"] == [sample_offer]


class TestGetRecentOffers:
    def test_get_recent_offers(self, mock_chromadb_collection):
        mock_chromadb_collection.get.return_value = {
            "metadatas": [{"address": "Test", "number_of_rooms": 3, "subways": True}]
        }
        result = get_recent_offers(
            mock_chromadb_collection, cutoff_time=1699999000.0, number_of_rooms=2
        )
        assert len(result) == 1
        mock_chromadb_collection.get.assert_called_once()
        call_kwargs = mock_chromadb_collection.get.call_args
        assert "where" in call_kwargs.kwargs
