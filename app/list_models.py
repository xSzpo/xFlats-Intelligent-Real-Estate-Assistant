#!/usr/bin/env python3
"""List available Gemini models and their capabilities."""

import os
from google import genai
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

print("Available Gemini Models:")
print("=" * 80)

try:
    # List all models
    models = client.models.list()

    for model in models:
        print(f"\nModel: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")

        # Check supported generation methods
        supported_methods = []
        if hasattr(model, "supported_generation_methods"):
            supported_methods = model.supported_generation_methods

        print(
            f"  Supported Methods: {', '.join(supported_methods) if supported_methods else 'N/A'}"
        )

        # Input/output token limits
        if hasattr(model, "input_token_limit"):
            print(f"  Input Token Limit: {model.input_token_limit:,}")
        if hasattr(model, "output_token_limit"):
            print(f"  Output Token Limit: {model.output_token_limit:,}")

        print("-" * 80)

except Exception as e:
    print(f"Error listing models: {e}")
    print("\nTrying alternative method...")

    # Try using the models endpoint directly
    try:
        response = client.models.list()
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
