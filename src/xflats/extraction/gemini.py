"""Gemini AI extraction for structured property data."""

import json
import re
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.api_core import retry
from google.genai import types
from pydantic import BaseModel

# Prompt templates
PROMPT_TEMPLATE = """
You are an expert in extracting apartment listings from cleaned HTML text. Your task is to extract key structured information and present it in **valid JSON format**.

Please follow these instructions **precisely**:

1. **Translate all text to English**, except for the **Address**, which must remain in its original language.
2. **Description**: in English, craft a neutral, informative overview covering:
  - Flat layout and standout positives/negatives
  - Natural light (e.g. "bright, east-facing")
  - Condition (e.g. "newly built", "recently renovated", "well-maintained older building")
  - View (e.g. "courtyard", "street-facing with greenery")
  - Neighborhood vibe (e.g. "quiet residential", "central and well-connected")
3. **Address**: Extract in this format: `Street Name Number, PostalCode City, Country`
   - Do NOT include unit/floor/apartment numbers in the address
4. **Price**: Extract as an integer, no commas or currency signs (e.g., `3250000`). If missing, use `null`.
5. **Area (m2)**: Extract as an integer (e.g., `87`). If missing, use `null`.
6. **Number of Rooms**: Extract total number of rooms as an integer. If missing, use `null`.
7. **Year Built**: Extract the year the building was constructed (e.g., `2006`). If missing, use `null`.
8. **Energy Label**: Extract as a single uppercase letter (`A`, `B`, etc.). If not available, use `null`.
9. **Balcony**: Return `true` if a balcony or terrace is mentioned; otherwise, `false`.
10. **URL**: Extract the full link to the listing.

Ensure the output is **JSON only**, with no explanation or additional text.

Cleaned HTMLs:

{html_content}

JSON output:
"""

SOURCE_TEMPLATE = """
---------------------
Offer #{i}
URL: {url}
SOURCE:
{text}
"""

EXAMPLE_TEXT = """
```json
[
    {
        "address": "Engholmene, 2450 København SV, Denmark",
        "description": "Apartment boasts abundant natural light and a spacious west-facing balcony overlooking the canal and marina. The contemporary interior is move-in ready, featuring high-quality materials. The neighborhood offers plenty of greenery, cafés, promenades, and convenient metro access.",
        "floor": "5",
        "price": 6195000,
        "area_m2": 91,
        "number_of_rooms": 2,
        "year_built": 2019,
        "energy_label": "A",
        "balcony": "true",
        "url": "https://www.boligsiden.dk/adresse/engholmene-2450-koebenhavn-sv-eksempel"
    }
]
"""


class Offers(BaseModel):
    """Pydantic model for a single property offer.

    Attributes:
        address: Street address in original language.
        description: English-language description of the property.
        floor: Floor number, if available.
        price: Listing price in local currency as integer, or ``None``.
        area_m2: Living area in square metres, or ``None``.
        number_of_rooms: Total room count, or ``None``.
        year_built: Construction year, or ``None``.
        energy_label: Single uppercase letter energy rating, or ``None``.
        balcony: Whether a balcony or terrace is present.
        url: Full URL to the original listing.
    """

    address: str
    description: str
    floor: str | None = None
    price: int | None = None
    area_m2: int | None = None
    number_of_rooms: int | None = None
    year_built: int | None = None
    energy_label: str | None = None
    balcony: bool
    url: str


class ListOfOffers(BaseModel):
    """Container for multiple property offers.

    Attributes:
        offers: List of extracted property offers.
    """

    offers: list[Offers]


def is_retriable(e: Exception) -> bool:
    """Determine whether a Gemini API error is retriable.

    Args:
        e: The exception raised during an API call.

    Returns:
        ``True`` if the error is a 429 or 503 API error, ``False`` otherwise.
    """
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


def fix_json(json_result: str) -> list[dict[str, Any]]:
    """Extract JSON objects from a malformed JSON string.

    Falls back to regex extraction when the Gemini response is not valid JSON.

    Args:
        json_result: Raw string that may contain embedded JSON objects.

    Returns:
        A list of parsed dictionaries, one per matched JSON object.
    """
    result = re.sub(r"(\n +)", "", json_result)
    pattern = r"(\{[^{}]+\})"
    matches = re.findall(pattern, result)
    return [json.loads(match) for match in matches]


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Google's Gemini API.

    Wraps the Gemini ``text-embedding-004`` model for use with ChromaDB.
    Toggle ``document_mode`` to switch between document and query embeddings.

    Attributes:
        client: Authenticated Gemini API client.
        document_mode: When ``True`` (default), produces document embeddings;
            when ``False``, produces query embeddings.
    """

    def __init__(self, client: genai.Client, *args: Any, **kwargs: Any) -> None:
        """Initialise the embedding function.

        Args:
            client: An authenticated ``genai.Client`` instance.
            *args: Positional arguments forwarded to the parent class.
            **kwargs: Keyword arguments forwarded to the parent class.
        """
        self.client = client
        self.document_mode = True
        super().__init__(*args, **kwargs)

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents or queries.

        Args:
            input: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input string.
        """
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def process_offers_with_ai(
    client: genai.Client, offers_source: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Process offer sources using Gemini AI to extract structured data.

    Sends cleaned HTML content to Gemini for structured extraction, falling
    back to regex-based JSON parsing on malformed responses.

    Args:
        client: An authenticated ``genai.Client`` instance.
        offers_source: List of dicts each containing ``url`` and ``text`` keys
            representing a single property listing.

    Returns:
        A list of dictionaries, each containing extracted property fields.
    """
    if not offers_source:
        return []

    source_content = ""
    for i, offer in enumerate(offers_source, 1):
        source_content += SOURCE_TEMPLATE.format(i=i, **offer)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ListOfOffers,
            max_output_tokens=65536,
        ),
        contents=[
            PROMPT_TEMPLATE.format(html_content=source_content),
            EXAMPLE_TEXT,
        ],
    )

    try:
        parsed = json.loads(response.text)
        if isinstance(parsed, dict) and "offers" in parsed:
            return list(parsed["offers"])
        if isinstance(parsed, list):
            return list(parsed)
    except json.JSONDecodeError:
        pass

    return fix_json(response.text) or []
