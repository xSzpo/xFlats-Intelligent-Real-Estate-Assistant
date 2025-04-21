import datetime
import hashlib
import json
import re
import statistics
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import boto3
import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.api_core import retry
from google.genai import types
from osmnx import features_from_point
from pydantic import BaseModel


def get_secret(secret_id, key=None, profile_name=None):
    if profile_name:
        boto3.setup_default_session(profile_name=profile_name)
    secrets_client = boto3.client("secretsmanager", region_name="eu-central-1")
    secret_value_response = secrets_client.get_secret_value(SecretId=secret_id)
    secret_dict = json.loads(secret_value_response["SecretString"])
    if key:
        return secret_dict[key]
    else:
        return secret_dict


client = genai.Client(
    api_key=get_secret(
        secret_id="gemini-274181059559", key="GOOGLE_API_KEY", profile_name="priv"
    )
)


def check_crawl_permission(target_page: str) -> bool:
    """
    Check if crawling is allowed for a given URL based on the site's robots.txt rules.

    Parameters:
        target_page (str): The URL of the page to check.

    Returns:
        bool: True if crawling is allowed, False if disallowed by robots.txt.
    """
    parsed = urlparse(target_page)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or "/"
    robots_url = urljoin(base_url, "/robots.txt")

    try:
        with urlopen(robots_url, timeout=5) as response:
            robots_content = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error fetching {robots_url}: {e}")
        return True

    rules: list[tuple[str, str]] = []
    current_agent = None

    for line in robots_content.splitlines():
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue

        field, value = [part.strip() for part in line.split(":", 1)]
        field = field.lower()

        if field == "user-agent":
            current_agent = value
        elif current_agent == "*" and field in ("allow", "disallow"):
            rules.append((field, value))

    best_match: str | None = None
    best_length = -1
    for directive, pattern in rules:
        if not pattern:
            continue
        regex = "^" + re.escape(pattern).replace("\\*", ".*")
        if re.search(regex, path):
            if len(pattern) > best_length:
                best_match = directive
                best_length = len(pattern)
            elif len(pattern) == best_length and directive == "allow":
                best_match = directive

    return best_match != "disallow"


def fetch_html(url: str) -> str:
    """
    Fetch HTML content from the specified URL.

    Parameters:
        url (str): The URL to fetch HTML from.

    Returns:
        str: The HTML content of the page.

    Raises:
        Exception: If the page retrieval fails.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    raise Exception(f"Failed to retrieve page. Status code: {response.status_code}")


class Offers(BaseModel):
    """
    Data model representing a property offer.
    """

    address: str
    lat: float
    long: float
    price: int
    area_m2: int
    number_of_rooms: int
    year_built: int
    energy_label: str
    url: str


class ListOfOffers(BaseModel):
    """
    Data model representing a list of property offers.
    """

    offers: list[Offers]


def is_retriable(e: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.

    Parameters:
        e (Exception): The exception to check.

    Returns:
        bool: True if the exception is retriable (API error codes 429 or 503), otherwise False.
    """
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


EXAMPLE_TEXT = """
EXAMPLE:
JSON Response:
```
[{
"address": "Ålekistevej 172, 2720 Vanløse",
"lat": 55.671677,
"long":12.553303,
"price": 2798000,
"area_m2": 87,
"number_of_rooms":3,
"year_built": 1988,
"energy_label": "C",
"url": "https://www.boligsiden.dk/adresse/aalekistevej-172-3-31-2720-vanloese-01018672_172__3__31",
}
]
```
"""


PROMPT_TEMPLATE = """
You are an expert in extracting property listings. Based on the HTML below from a real estate 
webpage, please extract all apartment offers into a valid JSON list using fewer than {max_tokens} 
tokens. The offers are in Danish; please translate the data into English. 

Please generate the URL of the offers using the pattern 'https://www.boligsiden.dk/adresse/<partial_url>', 
e.g. 'https://www.boligsiden.dk/adresse/aalekistevej-172-3-31-2720-vanloese-01018672_172__3__31'
webpage:\n\n

{html_content}

Output:
"""


def fix_json(json_result: str) -> list[dict]:
    """
    Clean and extract JSON objects from a string containing JSON fragments.

    Parameters:
        json_result (str): The raw JSON string output.

    Returns:
        list[dict]: A list of parsed JSON dictionaries.
    """
    result = re.sub(r"(\n +)", "", json_result)
    pattern = r"(\{[^{}]+\})"
    matches = re.findall(pattern, result)
    return [json.loads(match) for match in matches]


def get_public_transport_station(lat: float, long: float, dist: int = 700) -> str:
    """
    Retrieve public transport station information within a specified distance of a location.

    Parameters:
        lat (float): Latitude coordinate.
        long (float): Longitude coordinate.
        dist (int): Search radius in meters (default is 700).

    Returns:
        str: A semicolon-separated string of public transport stations, or an empty string if an error occurs.
    """
    try:
        point = (lat, long)
        tags = {"public_transport": "station"}
        pois = features_from_point(point, tags, dist=dist).reset_index()
        cols = ["name", "station", "amenity"]
        pois = pois.loc[:, pois.columns.isin(cols)].to_dict(orient="split")["data"]
        poi_set = {", ".join(filter(lambda x: x == x, item)) for item in pois}
        return "; ".join(poi_set)
    except Exception as e:
        print(e)
        return ""


def summarize_webpage(
    url: str, prompt: str, example: str, max_tokens: int = 8192
) -> list[dict]:
    """
    Extract and process property listings from a given webpage URL.

    This function fetches the webpage, generates a JSON response using the Gemini model,
    and enriches each listing with nearby public transport station data.

    Parameters:
        url (str): The webpage URL to process.
        prompt (str): The prompt template for content generation.
        example (str): An example text to guide output formatting.
        max_tokens (int): Maximum output tokens for the model (default is 8192).

    Returns:
        list[dict]: A list of property offer dictionaries.

    Raises:
        ValueError: If crawling is not permitted by robots.txt.
    """
    if not check_crawl_permission(url):
        raise ValueError(f"Crawling not permitted: {url}")

    html_content = fetch_html(url)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ListOfOffers,
            max_output_tokens=max_tokens,
        ),
        contents=[
            prompt.format(max_tokens=max_tokens, html_content=html_content),
            example,
        ],
    )
    results = fix_json(response.text)

    # Enrich each offer with public transport station data.
    for offer in results:
        offer["public_transport_station_up_to_700m"] = get_public_transport_station(
            offer["lat"], offer["long"]
        )
        offer["create_date"] = datetime.datetime.today().isoformat()
        offer["id"] = hashlib.shake_128(offer["url"].encode()).hexdigest(8)
    return results


def offer_to_text(offer: dict) -> str:
    """
    Convert a property offer dictionary into a descriptive text string.

    Parameters:
        offer (dict): The property offer data.

    Returns:
        str: A text representation of the offer.
    """
    return (
        f"Address: {offer.get('address', '')}. "
        f"Area: {offer.get('area_m2', '')} m². "
        f"Rooms: {offer.get('number_of_rooms', '')}. "
        f"Year built: {offer.get('year_built', '')}. "
        f"Energy label: {offer.get('energy_label', '')}. "
        f"Nearby transit (up to 700 m): {offer.get('public_transport_station_up_to_700m', '')}."
    )


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for generating text embeddings via the Gemini API.
    """

    document_mode: bool = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the provided documents.

        Parameters:
            input (Documents): The documents or queries to embed.

        Returns:
            Embeddings: A list of embedding vectors.
        """
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def setup_vector_database(ip=None, port=8000):
    """
    Initialize and set up ChromaDB for storing property embeddings.

    Returns:
        object: The initialized ChromaDB collection.
    """
    DB_NAME = "real-estate-offers"
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True
    if ip:
        chroma_client = chromadb.HttpClient(host=ip, port=port)
    else:
        chroma_client = chromadb.PersistentClient()

    collection = chroma_client.get_or_create_collection(
        name=DB_NAME, embedding_function=embed_fn
    )
    return collection


def add_offers_to_db(collection, offers: list[dict], batch_size: int = 100):
    """
    Add property offers to the vector database in batches.

    Parameters:
        collection: The ChromaDB collection.
        offers (list[dict]): List of property offer dictionaries.
        batch_size (int): Number of documents to add in each batch.
    """
    documents = [offer_to_text(offer) for offer in offers]
    ids = [offer["id"] for offer in offers]
    metadatas = offers

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"Added batch {i // batch_size + 1} with {len(batch_docs)} documents.")


def get_price_point(offer: dict, collection: any) -> float:
    """
    Calculate a price point for the offer based on similar offers in the collection.

    The function queries the collection for similar offers, computes an average price from the metadata,
    and divides the current offer price by the average.

    Parameters:
        offer (dict): The property offer.
        collection (any): The ChromaDB collection.

    Returns:
        float: The price ratio for the offer.
    """
    emb_results = collection.query(query_texts=[offer_to_text(offer)], n_results=5)[
        "metadatas"
    ][0]
    avg_price = statistics.mean([item["price"] for item in emb_results])
    return offer["price"] / avg_price if avg_price != 0 else 0.0


def get_similar_offers(offer: dict, collection: any) -> str:
    """
    Retrieve similar offers for a given property offer.

    Parameters:
        offer (dict): The property offer.
        collection (any): The ChromaDB collection.

    Returns:
        str: A text summary including the offer and similar offers.
    """
    emb_results = collection.query(query_texts=[offer_to_text(offer)], n_results=5)
    return f"Offer:\n\n{offer}\n\nSimilar offers:\n\n{emb_results}"


system_instruction_template = (
    "You are a helpful and informative bot that presents the user with the three best apartment sale offers "
    "of the day in Copenhagen. The apartments were chosen because their price was the lowest. "
    "We received five similar offers, calculated the average, and then divided the apartment price by the average. "
    "Please provide the best offers by including the address, price, size, number  of rooms, year built, public transport "
    "options, and the URL for each listing, along with a description of how each compares to similar offers."
    "Use only the information provided in the context.\n\n"
    "Best offers #1: {best_offer_1}\n\n"
    "Best offers #2: {best_offer_2}\n\n"
    "Best offers #3: {best_offer_3}\n\n"
)


def main():
    BASE_URL = (
        "https://www.boligsiden.dk/tilsalg/villa,ejerlejlighed?"
        "sortAscending=true&priceMax=7000000&polygon=12.626596,55.654896|"
        "12.641590,55.649906|12.636977,55.665739|12.602806,55.668303|"
        "12.584355,55.665330|12.585458,55.680916|12.586996,55.694136|"
        "12.566745,55.704787|12.556173,55.707496|12.535566,55.708713|"
        "12.523383,55.700403|12.513564,55.690885|12.530509,55.677542|"
        "12.527120,55.667137|12.538516,55.662200|12.545194,55.656305|"
        "12.560256,55.652160|12.573540,55.658583|12.590841,55.649690|"
        "12.602053,55.668091|12.622655,55.667008|12.626596,55.654896&sortBy=timeOnMarket&page={page}"
    )

    print("Starting data collection process")

    # Set up vector database
    collection = setup_vector_database(
        ip=get_secret(secret_id="chrome-db-274181059559", key="IP", profile_name="priv")
    )
    print("Vector database initialized")

    # Collect historical listings for context
    all_results = []
    # Process pages 2 to 5 for historical offers
    for page in range(1, 8):
        print(f"Processing historical listings from page {page}")
        page_url = BASE_URL.format(page=page)
        offers = summarize_webpage(page_url, PROMPT_TEMPLATE, EXAMPLE_TEXT)
        all_results.extend(offers)

    print(f"I have total {len(all_results)} historical listings")
    seen = set()
    all_results_unique = []
    for item in all_results:
        if item["id"] not in seen:
            seen.add(item["id"])
            all_results_unique.append(item)
    print(f"I have total {len(all_results_unique)} unique historical listings")

    # Add historical listings to vector database
    print(f"Adding {len(all_results_unique)} historical listings to vector database")
    add_offers_to_db(collection, all_results_unique)


if __name__ == "__main__":
    main()
