"""Shared test fixtures for xFlats test suite."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_offer():
    """Realistic offer dict matching the Offers Pydantic model."""
    return {
        "id": "test-offer-001",
        "address": "Vesterbrogade 42, 1620 København V, Denmark",
        "description": "Bright corner apartment with south-facing balcony overlooking courtyard. Recently renovated kitchen and bathroom. Quiet residential street close to metro.",
        "floor": "3",
        "price": 3250000,
        "area_m2": 87,
        "number_of_rooms": 3,
        "year_built": 1905,
        "energy_label": "C",
        "balcony": True,
        "url": "https://www.boligsiden.dk/adresse/vesterbrogade-42-1620-koebenhavn-v",
        "subways": True,
        "public_transport_text": "subways:Enghave Plads; subways:Vesterport;",
        "create_date": 1700000000.0,
    }


@pytest.fixture
def sample_html():
    """Minimal HTML containing /adresse/ links for scraper testing."""
    return """
    <html>
    <body>
        <a href="/adresse/vesterbrogade-42-1620-koebenhavn-v">Listing 1</a>
        <a href="/adresse/noerrebrogade-10-2200-koebenhavn-n?ref=search">Listing 2</a>
        <a href="/about">About page</a>
        <a href="/adresse/amagerbrogade-5-2300-koebenhavn-s">Listing 3</a>
    </body>
    </html>
    """


@pytest.fixture
def mock_chromadb_collection():
    """MagicMock mimicking chromadb Collection interface."""
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"price": 3000000}, {"price": 3500000}]],
        "distances": [[0.1, 0.2]],
    }
    collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
    collection.add.return_value = None
    return collection
