"""Tests for xflats.extraction.gemini."""

from unittest.mock import MagicMock

from xflats.extraction.gemini import GeminiEmbeddingFunction, Offers, fix_json


class TestOffersModel:
    def test_offers_model_validation(self):
        offer = Offers(
            address="Vesterbrogade 42, 1620 København V, Denmark",
            description="Bright apartment with balcony",
            floor="3",
            price=3250000,
            area_m2=87,
            number_of_rooms=3,
            year_built=1905,
            energy_label="C",
            balcony=True,
            url="https://www.boligsiden.dk/adresse/test",
        )
        assert offer.address == "Vesterbrogade 42, 1620 København V, Denmark"
        assert offer.price == 3250000
        assert offer.balcony is True
        assert offer.number_of_rooms == 3
        assert offer.floor == "3"

    def test_offers_model_optional_fields(self):
        offer = Offers(
            address="Test Address",
            description="Test",
            balcony=False,
            url="https://example.com",
        )
        assert offer.price is None
        assert offer.area_m2 is None
        assert offer.floor is None


class TestFixJson:
    def test_fix_json_valid(self):
        json_str = '[{"address": "Test", "price": 100}]'
        result = fix_json(json_str)
        assert len(result) == 1
        assert result[0]["address"] == "Test"

    def test_fix_json_with_markdown_wrapper(self):
        json_str = '```json\n{"address": "Test", "price": 200}\n```'
        result = fix_json(json_str)
        assert len(result) == 1
        assert result[0]["price"] == 200


class TestGeminiEmbeddingFunction:
    def test_gemini_embedding_function(self):
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[mock_embedding]
        )

        embed_fn = GeminiEmbeddingFunction(client=mock_client)
        result = embed_fn(["test document"])

        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args
        assert call_kwargs.kwargs["model"] == "models/text-embedding-004"
        assert call_kwargs.kwargs["contents"] == ["test document"]
        assert len(result) == 1
        assert list(result[0]) == [0.1, 0.2, 0.3]
