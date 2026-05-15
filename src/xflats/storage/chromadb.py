"""ChromaDB vector database operations."""

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

CHROMADB_DEFAULT_PORT = 8000
DB_NAME = "real-estate-offers"


def setup_vector_database(
    chromadb_ip: str, embed_fn: chromadb.EmbeddingFunction
) -> chromadb.Collection:
    """Initialize a ChromaDB collection with the given embedding function.

    Connects to a remote ChromaDB instance if ``chromadb_ip`` is provided,
    otherwise falls back to a local persistent client.

    Args:
        chromadb_ip: Hostname or IP of the ChromaDB server. When empty, a
            local ``PersistentClient`` is used instead.
        embed_fn: Embedding function passed to the collection for automatic
            document vectorisation.

    Returns:
        The ChromaDB collection ready for queries and inserts.
    """
    logger.info("Initializing vector database...")
    embed_fn.document_mode = True

    if chromadb_ip:
        chroma_client = chromadb.HttpClient(
            host=chromadb_ip, port=CHROMADB_DEFAULT_PORT
        )
    else:
        chroma_client = chromadb.PersistentClient()

    collection = chroma_client.get_or_create_collection(
        name=DB_NAME, embedding_function=embed_fn
    )
    logger.info("Vector database initialized")
    return collection


def check_if_document_exists(
    doc_id: str, collection: chromadb.Collection
) -> bool:
    """Check whether a document with the given ID exists in the collection.

    Args:
        doc_id: Unique document identifier to look up.
        collection: ChromaDB collection to query.

    Returns:
        ``True`` if the document exists, ``False`` otherwise.
    """
    result = collection.get(ids=[doc_id])
    return bool(result["ids"])


def add_offers_to_db(
    collection: chromadb.Collection,
    offers: list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """Add real-estate offers to the vector database in batches.

    Each offer is converted to a text representation via :func:`offer_to_text`
    before being inserted.

    Args:
        collection: ChromaDB collection to insert into.
        offers: List of offer dictionaries with at least an ``"id"`` key.
        batch_size: Maximum number of documents per insert call.
    """
    documents = [offer_to_text(offer) for offer in offers]
    ids = [str(offer.get("id", "")) for offer in offers]
    metadatas = offers

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)  # type: ignore[arg-type]
        logger.info(
            "Added batch %d with %d documents.",
            i // batch_size + 1,
            len(batch_docs),
        )


def offer_to_text(offer: dict[str, Any]) -> str:
    """Convert an offer dictionary to a human-readable text representation.

    The resulting string is used as the document body for vector embedding.

    Args:
        offer: Offer dictionary containing property details.

    Returns:
        A single-string summary of the offer's key attributes.
    """
    return (
        f"Address: {offer.get('address', '')}. "
        f"Description: {offer.get('description', '')}. "
        f"Floor: {offer.get('floor', '')}. "
        f"Area: {offer.get('area_m2', '')} m². "
        f"Rooms: {offer.get('number_of_rooms', '')}. "
        f"Balcony: {offer.get('balcony', '')}. "
        f"Year built: {offer.get('year_built', '')}. "
        f"Energy label: {offer.get('energy_label', '')}. "
        f"Nearby transit (up to 700 m): {offer.get('public_transport_text', '')}."
    )


def get_price_point(
    offer: dict[str, Any], collection: chromadb.Collection, n_results: int = 5
) -> float:
    """Compute the relative price point of an offer vs. similar recent listings.

    Queries the vector database for the ``n_results`` most similar offers
    created in the last 90 days and returns the ratio of the offer's price to
    the average price of those results.

    Args:
        offer: Offer dictionary that must contain a ``"price"`` key.
        collection: ChromaDB collection to query for similar offers.
        n_results: Number of similar offers to retrieve.

    Returns:
        Ratio of the offer price to the mean price of similar offers, or
        ``0.0`` when no comparable prices are found.
    """
    now = datetime.now(timezone.utc)
    emb_results = collection.query(
        include=["metadatas"],
        where={"create_date": {"$gt": (now - timedelta(days=90)).timestamp()}},
        query_texts=[offer_to_text(offer)],
        n_results=n_results,
    )["metadatas"][0]
    prices = [
        item.get("price") for item in emb_results if item.get("price") is not None
    ]
    if not prices:
        return 0.0
    avg_price = statistics.mean(prices)
    return offer.get("price") / avg_price if avg_price != 0 else 0.0


def get_similar_offers(
    offer: dict[str, Any], collection: chromadb.Collection
) -> str:
    """Return a formatted string of the offer and its nearest neighbours.

    Args:
        offer: Offer dictionary to find similarities for.
        collection: ChromaDB collection to query.

    Returns:
        Human-readable string containing the original offer and the top-5
        similar offers from the database.
    """
    emb_results = collection.query(query_texts=[offer_to_text(offer)], n_results=5)
    return f"Offer:\n\n{offer}\n\nSimilar offers:\n\n{emb_results}"


def get_recent_offers(
    collection: chromadb.Collection,
    cutoff_time: float,
    number_of_rooms: int,
) -> list[dict[str, Any]]:
    """Retrieve recent offers matching room-count and transit criteria.

    Args:
        collection: ChromaDB collection to query.
        cutoff_time: Unix timestamp; only offers created after this time are
            returned.
        number_of_rooms: Minimum number of rooms required.

    Returns:
        List of offer metadata dictionaries matching the filters.
    """
    results = collection.get(
        include=["metadatas"],
        where={
            "$and": [
                {"create_date": {"$gt": cutoff_time}},
                {"subways": {"$eq": True}},
                {"number_of_rooms": {"$gte": number_of_rooms}},
            ]
        },
    )
    return list(results.get("metadatas") or [])
