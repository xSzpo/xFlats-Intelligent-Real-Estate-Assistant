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
    lat: float, lon: float, radius: int = 700, max_retries: int = 5
):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})[public_transport=station];
      node(around:{radius},{lat},{lon})[railway=subway_entrance];
      node(around:{radius},{lat},{lon})[railway=station];
    );
    out body;
    """

    backoff_time = 30

    station_types = {
        "ferry_terminals": "ferry_terminal",
        "light_rails": "light_rail",
        "subways": "subway",
        "bus_stations": "bus_station",
        "trains": "train",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(overpass_url, params={"data": query})
            response.raise_for_status()
            data = response.json()

            stations = {key: [] for key in station_types}

            for element in data["elements"]:
                tags = element.get("tags", {})
                station_tag = (
                    tags.get("station")
                    or tags.get("public_transport")
                    or tags.get("railway")
                )
                name = tags.get("name")

                if name:
                    for key, value in station_types.items():
                        if station_tag == value:
                            stations[key].append(name)

            public_transport_text = ""
            for key in stations:
                if stations[key]:
                    public_transport_text += (
                        "; ".join([f"{key}:{i}" for i in set(stations[key])]) + "; "
                    )
                    stations[key] = True
                else:
                    stations[key] = False

            stations["public_transport_text"] = public_transport_text

            return stations

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"Failed after {max_retries} attempts: {e}")
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
        print(f"Retrieved {len(results)} offers from crawler.")
        for offer in results:
            if bool(offer.get("url")):
                offer.update(
                    get_public_transport_stations(offer.get("lat"), offer.get("long"))
                )
                offer["create_date"] = datetime.datetime.today().timestamp()
                offer["id"] = hashlib.shake_128(offer.get("url").encode()).hexdigest(8)
                time.sleep(0.5)
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
