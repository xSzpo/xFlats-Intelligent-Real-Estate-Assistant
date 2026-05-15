"""Telegram notification delivery."""

import re
from typing import Any

import requests

from xflats.storage.chromadb import get_price_point


def create_offer_text(offer_dict: dict) -> str:
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


def send_telegram_notifications(
    offers: list[dict[str, Any]],
    collection,
    telegram_token: str,
    telegram_chat_id: str,
) -> None:
    """Send offer notifications via Telegram."""
    if not offers:
        print("No apartment listings to share on Telegram")
        return

    print(f"Sending {len(offers)} listings to Telegram")

    for offer in offers:
        offer["price_point"] = get_price_point(offer, collection)

    offers.sort(key=lambda x: x.get("price_point", 0))

    for offer in offers:
        offer_text = create_offer_text(offer)
        print(f"Sending: {offer_text}")

        telegram_url = (
            f"https://api.telegram.org/bot{telegram_token}/"
            f"sendMessage?chat_id={telegram_chat_id}&text={offer_text}"
        )

        try:
            response = requests.get(telegram_url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to send Telegram message: {e}")
