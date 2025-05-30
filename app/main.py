import datetime
import hashlib
import os
import time

import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.api_core import retry
from google.genai import types
from pydantic import BaseModel
from utils import (
    add_offers_to_db,
    check_crawl_permission,
    chromadb_check_if_document_exists,
    create_offer_text,
    extract_adresse_urls,
    fetch_and_preprocess,
    fetch_html,
    fix_json,
    geocode_address,
    get_price_point,
    get_public_transport_stations,
    get_secret,
    is_retriable,
    offer_to_text,
    remove_url_parameters,
)

# bot page
# https://api.telegram.org/bot{telegram_token}/getUpdates

profile_name = os.getenv("AWS_PROFILE", None)

chromadb_ip = os.getenv("CHROMADB_IP", "chromadb")


telegram_token = api_key = get_secret(
    secret_id="telegram-274181059559", key="TOKEN", profile_name=profile_name
)

telegram_chat_id = api_key = get_secret(
    secret_id="telegram-274181059559", key="CHAT_ID", profile_name=profile_name
)

genai_api_key = get_secret(
    secret_id="gemini-274181059559", key="GOOGLE_API_KEY", profile_name=profile_name
)

client = genai.Client(api_key=genai_api_key)

OFFER_VERSION = 2

PROMPT_TEMPLATE = """
You are an expert in extracting apartment listings from cleaned HTML text. Your task is to extract key structured information and present it in **valid JSON format**.

Please follow these instructions **precisely**:

1. **Translate all text to English**, except for the **Address**, which must remain in its original language.
2. **Create a detailed apartment description** based on the listing, covering:
   - Natural light: Is it bright, which directions (e.g., east-facing)?
   - Condition: Is it newly built, recently renovated, or older but well-maintained?
   - View: What can be seen from the apartment? (e.g., courtyard, street, green area)
   - Neighborhood: What is mentioned about the area? Is it calm, central, well-connected, or popular?
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


class Offers(BaseModel):
    address: str
    description: str
    floor: str
    price: int
    area_m2: int
    number_of_rooms: int
    year_built: int
    energy_label: str
    balcony: str
    url: str


class ListOfOffers(BaseModel):
    offers: list[Offers]


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


BASE_URL = (
    "https://www.boligsiden.dk/tilsalg/villa,ejerlejlighed?sortAscending=true"
    "&mapBounds=7.780294,54.501948,15.330305,57.896401&priceMax=7000000"
    "&polygon=12.555001,55.714439|12.544964,55.711152|12.535566,55.708713|12.523383,55.700403|"
    "12.513564,55.690885|12.507604,55.674192|12.508089,55.656840|12.521769,55.648585|"
    "12.534702,55.642731|12.564876,55.614388|12.591917,55.614270|12.599055,55.649692|"
    "12.605518,55.649361|12.615303,55.649093|12.628699,55.649335|12.641590,55.649906|"
    "12.636977,55.665739|12.626008,55.676732|12.636641,55.686489|12.654036,55.720127|"
    "12.602392,55.730897|12.555001,55.714439&page={page}"
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


def main():
    print("Starting data collection process")

    # Set up vector database
    collection = setup_vector_database(
        ip=chromadb_ip,
        client=client,
    )
    print("Vector database initialized")

    # Collect historical listings for context
    all_results = []

    MAX_RETRIES = 3  # Number of times to retry a page
    NUMBER_OF_PAGES_TO_OPEN = int(os.getenv("NUMBER_OF_PAGES_TO_OPEN", 3))
    NUMBER_OF_ROOMS = int(os.getenv("NUMBER_OF_ROOMS", 2))
    GET_OFFERS_FROM_X_LAST_MIN = 5

    for page in range(1, NUMBER_OF_PAGES_TO_OPEN + 1):
        print(f"Processing historical listings from page {page}")
        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            page_url = BASE_URL.format(page=page)
            print(page_url)
            try:
                if not check_crawl_permission(page_url):
                    raise ValueError(f"Crawling not permitted: {page_url}")
                html_content = fetch_html(page_url)
                urls = extract_adresse_urls(html_content)
                urls_hash = [
                    (url, hashlib.shake_128(str(url).encode()).hexdigest(8))
                    for url in urls
                ]
                new_urls: list = []
                for url, hash_id in urls_hash:
                    if not chromadb_check_if_document_exists(hash_id, collection):
                        new_urls += [url]

                if not new_urls:
                    print(f"No new URLs found on page {page}, skipping.")
                    success = True
                    break

                offers_source: list[dict] = []

                for url in new_urls:
                    text = fetch_and_preprocess(url, mode="two_requests")
                    if text:
                        offers_source += [{"url": url, "text": text}]

                SOURCE_TEMPLATE = """
                ---------------------
                Offer #{i}
                URL: {url}  
                SOURCE:
                {text}
                """

                SOURCE = ""

                for i, offer in enumerate(offers_source):
                    SOURCE += SOURCE_TEMPLATE.format(i=i + 1, **offer)

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=ListOfOffers,
                        max_output_tokens=8192,
                    ),
                    contents=[
                        PROMPT_TEMPLATE.format(max_tokens=8192, html_content=SOURCE),
                        EXAMPLE_TEXT,
                    ],
                )
                offers = fix_json(response.text)

                if offers:
                    print(f"Retrieved {len(offers)} offers from crawler.")

                    # First, clean URLs
                    for offer in offers:
                        if offer.get("url"):
                            offer["url"] = remove_url_parameters(offer.get("url"))
                        # Assign IDs now that URLs are cleaned
                        if offer.get("url"):
                            offer["id"] = hashlib.shake_128(
                                offer["url"].encode()
                            ).hexdigest(8)
                        offer["version"] = OFFER_VERSION

                        try:
                            address = offer.get("address")
                            lat, lon = geocode_address(address)
                            offer["lat"], offer["long"] = lat, lon

                            # Fetch stations around the new coordinates
                            stations = get_public_transport_stations(lat=lat, lon=lon)
                            offer.update(stations)
                            print(f"Retrieved location for {address}")
                        except Exception as e:
                            print(f"No geocode_address retrieved, {e}")

                        offer["create_date"] = datetime.datetime.today().timestamp()
                        time.sleep(1)

                all_results.extend(offers)
                success = True
            except BaseException as e:
                print(e)
                offers = []
                retries += 1
                time.sleep(retries)
                print(f"Error retrieving page {page}, retry {retries}/{MAX_RETRIES}")

        if not success:
            print(f"Failed to retrieve page {page} after {MAX_RETRIES} retries.")

    print(f"I have total {len(all_results)} historical listings")
    seen = set()
    all_results_unique = []
    for item in all_results:
        if item.get("id") not in seen:
            seen.add(item.get("id"))
            all_results_unique.append(item)
    print(f"I have total {len(all_results_unique)} unique historical listings")

    # Add historical listings to vector database
    print(f"Adding {len(all_results_unique)} historical listings to vector database")

    for offer in all_results_unique:
        print(offer_to_text(offer))

    add_offers_to_db(collection, all_results_unique)

    # get last offers
    now = datetime.datetime.now()

    newest_results = collection.get(
        include=["metadatas"],
        where={
            "$and": [
                {
                    "create_date": {
                        "$gt": (
                            now - datetime.timedelta(minutes=GET_OFFERS_FROM_X_LAST_MIN)
                        ).timestamp()
                    }
                },
                {"subways": {"$eq": True}},
                {"number_of_rooms": {"$gte": NUMBER_OF_ROOMS}},
            ]
        },
    )["metadatas"]

    if newest_results:
        print(
            f"Number of apartment listings to share on Telegram: {len(newest_results)}"
        )

        for offer in newest_results:
            offer["price_point"] = get_price_point(offer, collection)

        newest_results.sort(key=lambda x: x.get("price_point", 0))

        for offer in newest_results:
            offer_txt = create_offer_text(offer)
            print(offer_txt)
            url = (
                f"https://api.telegram.org/bot{telegram_token}/"
                f"sendMessage?chat_id={telegram_chat_id}&text={offer_txt}"
            )
            requests.get(url).json()

    else:
        print("There are no apartment listings to share on Telegram.")


if __name__ == "__main__":
    main()
