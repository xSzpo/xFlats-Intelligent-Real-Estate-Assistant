import datetime
import hashlib
import json
import re
import statistics
import time
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


def with_overpass_retry(fn, max_retries=3, retry_delay=60):
    """
    Retry a function that calls Overpass API, waiting `retry_delay` seconds after
    connection refused errors.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.ConnectionError as e:
            if "Connection refused" in str(e):
                print(
                    f"Overpass API refused connection. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            # For other errors (like HTTP 429, server errors), print and retry after delay
            print(
                f"Other Overpass error: {e}. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(retry_delay)
    raise Exception("Exceeded maximum retries to Overpass API")


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
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    raise Exception(f"Failed to retrieve page. Status code: {response.status_code}")


def is_retriable(e: Exception) -> bool:
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


def fix_json(json_result: str) -> list[dict]:
    result = re.sub(r"(\n +)", "", json_result)
    pattern = r"(\{[^{}]+\})"
    matches = re.findall(pattern, result)
    return [json.loads(match) for match in matches]


def get_public_transport_stations(
    lat: float, lon: float, dist: int = 700
) -> tuple[dict, str]:
    """
    Retrieve names of nearby public transport stations grouped by mode.
    Retries on Overpass errors.
    """
    try:
        point = (lat, lon)
        tags = {"public_transport": "station"}
        pois = with_overpass_retry(
            lambda: features_from_point(point, tags, dist=dist).reset_index()
        )
        if "amenity" in pois:
            pois.fillna({"station": pois["amenity"]}, inplace=True)

        def extract_names(column: str, value: str) -> str:
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
            if hasattr(stations[station], "size") and stations[station].size > 0:
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
    if not check_crawl_permission(url):
        raise ValueError(f"Crawling not permitted: {url}")

    class Offers(BaseModel):
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
        offers: list[Offers]

    html_content = fetch_html(url)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
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
    return (
        f"Address: {offer.get('address', '')}. "
        f"Area: {offer.get('area_m2', '')} m². "
        f"Rooms: {offer.get('number_of_rooms', '')}. "
        f"Year built: {offer.get('year_built', '')}. "
        f"Energy label: {offer.get('energy_label', '')}. "
        f"Nearby transit (up to 700 m): {offer.get('public_transport_text', '')}."
    )


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client, *args, **kwargs):
        self.client = client
        super().__init__(*args, **kwargs)

    document_mode: bool = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def setup_vector_database(ip: int, client: any, port=8000):
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
    emb_results = collection.query(query_texts=[offer_to_text(offer)], n_results=5)
    return f"Offer:\n\n{offer}\n\nSimilar offers:\n\n{emb_results}"


def create_offer_text(offer_dict) -> str:
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
