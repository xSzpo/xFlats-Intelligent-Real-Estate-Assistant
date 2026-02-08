#!/usr/bin/env python3
"""Check ChromaDB collection status and data."""

import chromadb

# Connect to ChromaDB (use 'chromadb' hostname in Docker network)
client = chromadb.HttpClient(host="chromadb", port=8000)

# List all collections
collections = client.list_collections()
print(f"Total collections: {len(collections)}")
for coll in collections:
    print(f"\nCollection: {coll.name}")
    print(f"  ID: {coll.id}")
    print(f"  Metadata: {coll.metadata}")
    
    # Get collection and count
    try:
        collection = client.get_collection(name=coll.name)
        count = collection.count()
        print(f"  Document count: {count}")
        
        if count > 0:
            print(f"  Has data - sampling first document...")
            # Get a sample
            results = collection.get(limit=1, include=["documents", "metadatas"])
            print(f"  Sample ID: {results['ids'][0] if results['ids'] else 'None'}")
            if results['metadatas']:
                print(f"  Sample metadata keys: {list(results['metadatas'][0].keys())}")
        else:
            print(f"  Collection is EMPTY - safe to delete and recreate")
    except Exception as e:
        print(f"  Error getting collection details: {e}")

print("\n" + "="*80)
if not collections:
    print("No collections found - fresh database")
else:
    print(f"Status: {sum(1 for c in collections)} collection(s) found")
