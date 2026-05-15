"""ChromaDB vector database operations."""

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb

CHROMADB_DEFAULT_PORT = 8000
DB_NAME = "real-estate-offers"


def setup_vector_database(chromadb_ip: str, embed_fn) -> chromadb.Collection:
    """Initialize ChromaDB collection with given embedding function."""
    print("Initializing vector database...")
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
    print("Vector database initialized")
    return collection


def check_if_document_exists(doc_id: str, collection: chromadb.Collection) -> bool:
    result = collection.get(ids=[doc_id])
    return bool(result["ids"])


def add_offers_to_db(
    collection: chromadb.Collection,
    offers: list[dict],
    batch_size: int = 100,
):
    documents = [offer_to_text(offer) for offer in offers]
    ids = [str(offer.get("id", "")) for offer in offers]
    metadatas = offers

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)  # type: ignore[arg-type]
        print(f"Added batch {i // batch_size + 1} with {len(batch_docs)} documents.")


def offer_to_text(offer: dict) -> str:
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


def get_price_point(offer: dict, collection, n_results: int = 5) -> float:
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


def get_similar_offers(offer: dict, collection) -> str:
    emb_results = collection.query(query_texts=[offer_to_text(offer)], n_results=5)
    return f"Offer:\n\n{offer}\n\nSimilar offers:\n\n{emb_results}"


def get_recent_offers(
    collection, cutoff_time: float, number_of_rooms: int
) -> list[dict[str, Any]]:
    """Retrieve recent offers matching criteria."""
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
