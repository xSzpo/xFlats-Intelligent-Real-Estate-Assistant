import datetime
import json
import re
import statistics
import time
from typing import List
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import boto3
import chromadb
import requests
from bs4 import BeautifulSoup, Comment
from google import genai
from pydantic import HttpUrl, ValidationError


def extract_adresse_urls(
    html_content: str, base_url: str = "https://www.boligsiden.dk"
) -> List[HttpUrl]:
    """
    Parse the given HTML and return a list of unique, validated HttpUrl objects
    for all links matching the pattern '/adresse/...'.
    Query parameters (e.g. '?udbud=...') are stripped off.

    :param html_content: HTML page source to scan for /adresse/ links
    :param base_url:     Base URL to resolve relative links against
    :return:             List of unique HttpUrl instances
    """

    def remove_url_parameters(path: str) -> str:
        return path.split("?", 1)[0]

    soup = BeautifulSoup(html_content, "html.parser")
    pattern = re.compile(r"^/adresse/")

    # Collect unique hrefs
    raw_paths = {tag["href"] for tag in soup.find_all("a", href=pattern)}

    validated_urls = set()
    for path in raw_paths:
        clean_path = remove_url_parameters(path)
        full = urljoin(base_url, clean_path)
        try:
            # Validate and canonicalize
            validated_urls.add(HttpUrl(full))
        except ValidationError:
            # Skip any malformed URLs
            continue

    return list(validated_urls)


def chromadb_check_if_document_exists(
    doc_id: str, collection: chromadb.api.models.Collection.Collection
) -> bool:
    result = collection.get(ids=[doc_id])
    return bool(result["ids"])


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


def preprocess_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Define elements to remove (excluding 'svg' to retain important energy labels)
    elements_to_remove = [
        "script",
        "style",
        "meta",
        "link",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "input",
        "button",
        "select",
        "option",
        "textarea",
        "canvas",
        "iframe",
        "noscript",
    ]

    # Remove defined elements
    for tag in soup(elements_to_remove):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    return str(soup)


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


def geocode_address(address: str) -> tuple[float, float]:
    """
    Use OSM Nominatim to turn a street address into (lat, lon).
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {
        "User-Agent": "my_app/1.0 (youremail@example.com)"  # ← set your own contact info
    }
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No location found for address: {address!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def get_public_transport_stations(
    lat: float = None,
    lon: float = None,
    address: str = None,
    radius: int = 700,
    max_retries: int = 5,
):
    # If an address was passed, geocode it first:
    if address is not None:
        lat, lon = geocode_address(address)

    if lat is None or lon is None:
        raise ValueError("Must supply either an address or both lat and lon")

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
                        "; ".join(f"{key}:{i}" for i in set(stations[key])) + "; "
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


def filter_unique_ids(dict_list, id_key="id"):
    seen_ids = set()
    result = []
    for item in dict_list:
        # remove objects without ID
        if item.get(id_key):
            unique_id = item.get(id_key)
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                result.append(item)
    return result


def _preprocess_html(html: str) -> str:
    """Core HTML cleanup: strip <head>, remove noise tags, keep JSON-LD, strip attrs, collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    # 1) Drop head
    if soup.head:
        soup.head.decompose()

    # 2) Strip comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # 3) Extract JSON-LD blocks
    jsonld = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        if tag.string:
            jsonld.append(tag.string.strip())
        tag.decompose()

    # 4) Remove boilerplate tags
    for name in (
        "script",
        "style",
        "meta",
        "link",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "input",
        "button",
        "select",
        "option",
        "textarea",
        "canvas",
        "iframe",
        "noscript",
    ):
        for t in soup.find_all(name):
            t.decompose()

    # 5) Strip *all* attributes
    for t in soup.find_all(True):
        t.attrs = {}

    # 6) Grab text and collapse whitespace
    text = (soup.body or soup).get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    # 7) Prepend JSON-LD
    for block in jsonld:
        text = f"<JSON-LD>{block}</JSON-LD> " + text

    return text


def fetch_and_preprocess(
    url: str, timeout: float = 5.0, max_bytes: int = 10_000, mode: str = "two_requests"
) -> str | None:
    """
    Fetch + preprocess a page if it exists. Returns cleaned text or None on 404/client-react-404/etc.

    Modes:
      - "two_requests":  streamed sniff GET → full GET → preprocess
      - "single_request": full streamed GET → sniff → preprocess
    """
    not_found_markers = [
        '<html id="__next_error__">',
        "NEXT_NOT_FOUND",
        "Siden findes ikke!",
    ]

    try:
        # -- Mode A: sniff first, then full fetch --
        if mode == "two_requests":
            resp = requests.get(url, timeout=timeout, stream=True)
            if 400 <= resp.status_code < 500:
                return None

            # sniff first max_bytes
            total = 0
            buf = []
            for chunk in resp.iter_content(1024, decode_unicode=True):
                buf.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            snippet = "".join(buf)
            resp.close()

            if any(marker in snippet for marker in not_found_markers):
                return None

            # full fetch
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            html = resp.text

        # -- Mode B: single streamed GET, full-buffered --
        elif mode == "single_request":
            resp = requests.get(url, timeout=timeout, stream=True)
            if 400 <= resp.status_code < 500:
                return None

            # read entire body
            buf = []
            for chunk in resp.iter_content(1024, decode_unicode=True):
                buf.append(chunk)
            resp.close()
            html = "".join(buf)

            # sniff for 404 markers
            if any(marker in html for marker in not_found_markers):
                return None

        else:
            raise ValueError("mode must be 'two_requests' or 'single_request'")

    except requests.RequestException:
        return None

    # preprocess and return
    return _preprocess_html(html)


def remove_url_parameters(url: str) -> str:
    return url.split("?", 1)[0]


def offer_to_text(offer: dict) -> str:
    return (
        f"Address: {offer.get('address', '')}. "
        f"Description: {offer.get('description', '')}. "
        f"Floor: {offer.get('floor', '')}. "
        f"Area: {offer.get('area_m2', '')} m². "
        f"Rooms: {offer.get('number_of_rooms', '')}. "
        f"Balcony: {offer.get('balcony', '')}. "
        f"Year built: {offer.get('year_built', '')}. "
        f"Energy label: {offer.get('energy_label', '')}. "
        f"Nearby transit (up to 700 m): {offer.get('public_transport_text', '')}."
    )


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
Description: {description}
Url: {url}
    """.format(**offer_dict, subways_txt=subways_txt)

    return offer_txt


def add_offers_to_db(
    collection: chromadb.api.models.Collection.Collection,
    offers: list[dict],
    batch_size: int = 100,
):
    documents = [offer_to_text(offer) for offer in offers]
    ids = [offer.get("id") for offer in offers]
    metadatas = offers

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"Added batch {i // batch_size + 1} with {len(batch_docs)} documents.")
