#!/usr/bin/env python3
"""Test which embedding model actually works with the Gemini API."""

import os
import sys
sys.path.insert(0, "/app")

from google import genai
from google.genai import types
from utils import get_secret

# Get API key
profile_name = os.getenv("AWS_PROFILE", None)
api_key = get_secret(
    secret_id="gemini-011337673661",
    key="GOOGLE_API_KEY",
    profile_name=profile_name,
)

# Initialize client
client = genai.Client(api_key=api_key)

# Test different embedding model names
embedding_models_to_test = [
    "text-embedding-004",
    "models/text-embedding-004",
    "gemini-embedding-001",
    "models/gemini-embedding-001",
    "embedding-001",
    "models/embedding-001",
]

test_text = ["This is a test apartment listing in Warsaw"]

print("Testing Embedding Models:")
print("=" * 80)

for model_name in embedding_models_to_test:
    print(f"\nTesting: {model_name}")
    try:
        response = client.models.embed_content(
            model=model_name,
            contents=test_text,
            config=types.EmbedContentConfig(task_type="retrieval_document"),
        )
        
        if response and response.embeddings:
            embedding_length = len(response.embeddings[0].values)
            print(f"  ✅ SUCCESS! Embedding dimension: {embedding_length}")
            print(f"  First 5 values: {response.embeddings[0].values[:5]}")
        else:
            print(f"  ❌ FAILED: No embeddings returned")
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            print(f"  ❌ FAILED: Model not found (404)")
        elif "429" in error_msg:
            print(f"  ❌ FAILED: Quota exceeded (429)")
        else:
            print(f"  ❌ FAILED: {error_msg[:100]}")

print("\n" + "=" * 80)
print("Test complete!")
