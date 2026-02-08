#!/usr/bin/env python3
"""Migrate ChromaDB collection to new embedding model while preserving data."""

import os
import sys
sys.path.insert(0, "/app")

import chromadb
from main import GeminiEmbeddingFunction
from google import genai
from utils import get_secret

print("=== ChromaDB Collection Migration ===")
print("This will migrate data from text-embedding-004 to gemini-embedding-001")
print()

# Get API key
profile_name = os.getenv("AWS_PROFILE", None)
api_key = get_secret(
    secret_id="gemini-011337673661",
    key="GOOGLE_API_KEY",
    profile_name=profile_name,
)

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Connect to ChromaDB
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# Step 1: Export existing data
print("Step 1: Exporting existing data...")
old_collection = chroma_client.get_collection(name="real-estate-offers-warsaw")
count = old_collection.count()
print(f"  Found {count} documents to migrate")

# Get all data in batches
all_ids = []
all_metadatas = []
all_documents = []

batch_size = 100
for offset in range(0, count, batch_size):
    limit = min(batch_size, count - offset)
    print(f"  Exporting batch {offset // batch_size + 1}: offset={offset}, limit={limit}")
    
    results = old_collection.get(
        limit=limit,
        offset=offset,
        include=["documents", "metadatas"]
    )
    
    all_ids.extend(results["ids"])
    all_metadatas.extend(results["metadatas"])
    all_documents.extend(results["documents"])

print(f"  Exported {len(all_ids)} documents")

# Step 2: Delete old collection
print("\nStep 2: Deleting old collection with broken embedding model...")
chroma_client.delete_collection(name="real-estate-offers-warsaw")
print("  Old collection deleted")

# Step 3: Create new collection with working embedding model
print("\nStep 3: Creating new collection with gemini-embedding-001...")
embed_fn = GeminiEmbeddingFunction(client)
embed_fn.document_mode = True

new_collection = chroma_client.create_collection(
    name="real-estate-offers-warsaw",
    embedding_function=embed_fn
)
print("  New collection created")

# Step 4: Re-import data (will re-embed with new model)
print("\nStep 4: Re-importing data with new embeddings...")
print("  This will take time as each document needs to be re-embedded...")

# Import in batches to avoid quota issues
batch_size = 10  # Smaller batches to avoid rate limits
for i in range(0, len(all_ids), batch_size):
    batch_ids = all_ids[i:i+batch_size]
    batch_docs = all_documents[i:i+batch_size]
    batch_meta = all_metadatas[i:i+batch_size]
    
    print(f"  Importing batch {i // batch_size + 1}/{(len(all_ids) + batch_size - 1) // batch_size} ({len(batch_ids)} docs)")
    
    try:
        new_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
    except Exception as e:
        print(f"  ⚠️  Error in batch {i // batch_size + 1}: {e}")
        print(f"  Continuing with next batch...")
        continue

final_count = new_collection.count()
print(f"\n✅ Migration complete!")
print(f"  Original documents: {count}")
print(f"  Migrated documents: {final_count}")

if final_count == count:
    print("  ✅ All documents successfully migrated")
else:
    print(f"  ⚠️  Warning: {count - final_count} documents missing")
