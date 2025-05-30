import datetime
import os
import time

import requests
from google import genai
from utils import (
    add_offers_to_db,
    create_offer_text,
    get_price_point,
    get_secret,
    setup_vector_database,
    summarize_webpage,
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

PROMPT_TEMPLATE = """
You are an expert in extracting property listings from HTML. Your task is to precisely extract
the following information about apartments for sale and represent the data in a JSON format.
You *must* follow these *exact* instructions:

1.  **Extract only apartment listings (Ejerlejlighed) for sale.**
2.  **Address:** Extract the full address in the format `Street Name Number, PostalCode City, Country`,
    e.g., `Vestergade 10, 1451 København K, Denmark`. If the floor/unit is mentioned in address, do not include it.
3.  **Price:** Extract the price as an integer (e.g., `2798000`). If the price is not found, set value as `None`.
4.  **Area (m2):** Extract the apartment area in square meters as an integer (e.g., `87`). If the area is not found, set value as `None`.
5.  **Number of Rooms:** Extract number of rooms as integer. If the rooms are not found, set value as `None`.
6.  **Year Built:** Extract the year the building was built as an integer (e.g., 1988). If not available, use `None`.
7.  **Energy Label:** Extract energy label (e.g., 'A', 'B', 'C'). If not available, set value as `None`.
8.  **URL:** Generate the URL for each listing using the pattern 'https://www.boligsiden.dk/adresse/<partial_url>' where the partial URL must match
     the actual URL from the page. Make sure that offer url matches the offer and the address. Do not include any parameters after `?`
9.  **If you find multiple URLs, make sure each URL belongs to its apartment's details.**
10. **If the field has no values, use `None`**.
11. **If the price is in the format of X.XXX.XXX, remove the '.' separators and use the number.**
12. Do not include any additional text or notes in the JSON, only the JSON is desired.

HTML:\n\n

{html_content}

JSON output:
"""


EXAMPLE_TEXT = """
```json
[
    {
        "address": "Vestergade 10, 1451 København K, Denmark",
        "price": 2798000,
        "area_m2": 87,
        "number_of_rooms": 3,
        "year_built": 1988,
        "energy_label": "C",
        "url": "https://www.boligsiden.dk/adresse/aalekistevej-172-3-31-2720-vanloese-01018672_172__3__31",
    },
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
                offers = summarize_webpage(
                    page_url, PROMPT_TEMPLATE, EXAMPLE_TEXT, client
                )
            except BaseException:
                offers = []

            if len(offers) > 5:
                all_results.extend(offers)
                success = True
            else:
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
            url = (
                f"https://api.telegram.org/bot{telegram_token}/"
                f"sendMessage?chat_id={telegram_chat_id}&text={offer_txt}"
            )
            requests.get(url).json()

    else:
        print("There are no apartment listings to share on Telegram.")


if __name__ == "__main__":
    main()
