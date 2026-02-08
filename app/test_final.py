#!/usr/bin/env python3
"""Final test to verify gemini-embedding-001 works end-to-end."""

import os
import sys
sys.path.insert(0, "/app")

import chromadb
from main import GeminiEmbeddingFunction
from google import genai
from utils import get_secret

print("=== Final Embedding Test ===\n")

# Step 1: Test embedding model directly
print("Step 1: Testing gemini-embedding-001 directly...")
profile_name = os.getenv("AWS_PROFILE", None)
api_key = get_secret(
    secret_id="gemini-011337673661",
    key="GOOGLE_API_KEY",
    profile_name=profile_name,
)

client = genai.Client(api_key=api_key)
embed_fn = GeminiEmbeddingFunction(client)
embed_fn.document_mode = True

test_doc = "Beautiful 3-room apartment in Warsaw centrum, 65m2, modern kitchen"
try:
    embeddings = embed_fn([test_doc])
    print(f"  ✅ SUCCESS! Generated {len(embeddings)} embedding(s)")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    print(f"  First 3 values: {embeddings[0][:3]}\n")
except Exception as e:
    print(f"  ❌ FAILED: {e}\n")
    sys.exit(1)

# Step 2: Test ChromaDB with the embedding function
print("Step 2: Testing ChromaDB with gemini-embedding-001...")
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# Get existing collection
try:
    collection = chroma_client.get_collection(
        name="real-estate-offers-warsaw",
        embedding_function=embed_fn
    )
    count = collection.count()
    print(f"  ✅ Connected to existing collection with {count} documents\n")
except Exception as e:
    print(f"   ⚠️  Could not get existing collection: {e}")
    print(f"  Creating new collection...")
    try:
        collection = chroma_client.create_collection(
            name="real-estate-offers-warsaw",
            embedding_function=embed_fn
        )
        print(f"  ✅ Created new collection\n")
    except Exception as e:
        print(f"  ❌ FAILED to create collection: {e}\n")
        sys.exit(1)

# Step 3: Test adding a document
print("Step 3: Testing document add with embedding...")  
test_id = "test-embedding-verification"
test_meta = {
    "address": "Test Street, Warsaw",
    "price": 500000,
    "number_of_rooms": 3,
    "area_m2": 65
}

try:
    collection.add(
        ids=[test_id],
        documents=[test_doc],
        metadatas=[test_meta]
    )
    print(f"  ✅ Successfully added test document with embedding\n")
except Exception as e:
    print(f"  ❌ FAILED to add document: {e}\n")
    sys.exit(1)

# Step 4: Test semantic search
print("Step 4: Testing semantic search...")
try:
    results = collection.query(
        query_texts=["apartment near city center"],
        n_results=3
    )
    print(f"  ✅ Search returned {len(results['ids'][0])} results")
    if results['ids'][0]:
        print(f"  Top result ID: {results['ids'][0][0]}")
        print(f"  Top result distance: {results['distances'][0][0]:.4f}\n")
except Exception as e:
    print(f"  ❌ FAILED to search: {e}\n")
    sys.exit(1)

# Cleanup
try:
    collection.delete(ids=[test_id])
    print("  Cleaned up test document\n")
except:
    pass

print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("gemini-embedding-001 is working correctly with ChromaDB")
print("=" * 60)
