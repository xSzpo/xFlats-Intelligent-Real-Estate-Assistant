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


def is_retriable(e: Exception) -> bool:
    """
    Determine if an exception should trigger a retry.

    Parameters:
        e (Exception): The exception to check.

    Returns:
        bool: True if the exception is retriable (API error codes 429 or 503), otherwise False.
    """
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


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


def get_public_transport_stations(
    lat: float, lon: float, dist: int = 700
) -> tuple[dict, str]:
    """
    Retrieve names of nearby public transport stations grouped by mode.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        dist (int): Search radius in meters (default is 700).

    Returns:
        Dictionary with five text fields (station names joined by comma) for:
        ferry_terminals, light_rails, subways, bus_stations, trains.
        If no station is found, returns "no_name" for that mode.
    """
    try:
        point = (lat, lon)
        tags = {"public_transport": "station"}
        # Fetch all relevant public transport stations
        pois = features_from_point(point, tags, dist=dist).reset_index()
        if "amenity" in pois:
            pois.fillna({"station": pois["amenity"]}, inplace=True)

        def extract_names(column: str, value: str) -> str:
            """Extract unique names where column == value, joined into text."""
            if column not in pois.columns:
                return []
            filtered = pois[pois[column] == value]["name"].dropna().unique()
            return filtered

        stations = {
            "ferry_terminals": extract_names("station", "ferry_terminal"),
            "light_rails": extract_names("station", "light_rail"),
            "subways": extract_names("station", "subway"),
            "bus_stations": extract_names("station", "bus_station"),
            "trains": extract_names("station", "train"),
        }
        text = ""

        for station in stations:
            if stations[station].size > 0:
                text += "; ".join([f"{station}:{i}" for i in stations[station]])
                text += "; "
                stations[station] = True
            else:
                stations[station] = False

        stations.update({"public_transport_text": text})

        return stations

    except Exception as e:
        print(e)
        return {
            "ferry_terminals": False,
            "light_rails": False,
            "subways": False,
            "bus_stations": False,
            "trains": False,
            "public_transport_text": "",
        }


def summarize_webpage(
    url: str,
    prompt: str,
    example: str,
    client: any,
    max_tokens: int = 8192,
) -> list[dict] | None:
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

    if results:
        for offer in results:
            if bool(offer.get("url")):
                offer.update(
                    get_public_transport_stations(offer.get("lat"), offer.get("long"))
                )
                offer["create_date"] = datetime.datetime.today().timestamp()
                offer["id"] = hashlib.shake_128(offer.get("url").encode()).hexdigest(8)
        return results
    else:
        return None


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
        f"Nearby transit (up to 700 m): {offer.get('public_transport_text', '')}."
    )


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for generating text embeddings via the Gemini API.
    """

    def __init__(self, client, *args, **kwargs):
        self.client = client
        super().__init__(*args, **kwargs)

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
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def setup_vector_database(ip: int, client: any, port=8000):
    """
    Initialize and set up ChromaDB for storing property embeddings.

    Returns:
        object: The initialized ChromaDB collection.
    """
    DB_NAME = "real-estate-offers"
    embed_fn = GeminiEmbeddingFunction(client)
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
    ids = [offer.get("id") for offer in offers]
    metadatas = offers

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"Added batch {i // batch_size + 1} with {len(batch_docs)} documents.")


def get_price_point(offer: dict, collection: any, n_results=5) -> float:
    """
    Calculate a price point for the offer based on similar offers in the collection.

    The function queries the collection for similar offers, computes an average price from the metadata,
    and divides the current offer price by the average.

    Parameters:
        offer (dict): The property offer.
        collection (any): The ChromaDB collection.
        n_results (int): number of similar offers to retrieve.

    Returns:
        float: The price ratio for the offer.
    """
    now = datetime.datetime.now()

    emb_results = collection.query(
        include=["metadatas"],
        where={"create_date": {"$gt": (now - datetime.timedelta(days=90)).timestamp()}},
        query_texts=[offer_to_text(offer)],
        n_results=n_results,
    )["metadatas"][0]
    avg_price = statistics.mean([item.get("price") for item in emb_results])
    return offer.get("price") / avg_price if avg_price != 0 else 0.0


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


def create_offer_text(offer_dict) -> str:
    """
    Build a multi-line description of a property offer, optionally extracting
    subway names from the raw transport text.

    Extracts any occurrences of 'subways:...;' if `offer_dict['subways']` is True,
    joins them with commas, and injects all values into the template.

    Returns:
        A formatted multi-line string including all fields and the
        comma-separated list of subways (if any).
    """
    if offer_dict.get("subways", False):
        pattern = re.compile(r"(?<=subways:)([^;]+)(?=;)")
        subways = pattern.findall(offer_dict["public_transport_text"])
        subways_txt = ", ".join(subways)
    else:
        subways_txt = ""

    offer_txt = """
Address: {address}
Size: {area_m2} m2, Rooms: {number_of_rooms}, Year: {year_built}, Energy: {energy_label}
Price: {price:,} DKK ({price_point:.2%})
Subway(s): {subways_txt}
Url: {url}
    """.format(**offer_dict, subways_txt=subways_txt)

    return offer_txt
