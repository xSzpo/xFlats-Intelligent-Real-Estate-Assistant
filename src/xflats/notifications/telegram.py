"""Telegram notification delivery."""

import logging
import re
from typing import Any

import chromadb
import requests

from xflats.storage.chromadb import get_price_point

logger = logging.getLogger(__name__)


def create_offer_text(offer_dict: dict[str, Any]) -> str:
    """Create formatted text for a single apartment offer.

    Args:
        offer_dict: Dictionary containing offer details such as address,
            price, area, and public transport information.

    Returns:
        Formatted string with offer details ready for Telegram delivery.
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
Description: {description}
Url: {url}
    """.format(**offer_dict, subways_txt=subways_txt)

    return offer_txt


def send_telegram_notifications(
    offers: list[dict[str, Any]],
    collection: chromadb.Collection,
    telegram_token: str,
    telegram_chat_id: str,
) -> None:
    """Send offer notifications via Telegram.

    Args:
        offers: List of offer dictionaries containing apartment listing data.
        collection: ChromaDB collection used for price point lookups.
        telegram_token: Telegram Bot API token.
        telegram_chat_id: Target Telegram chat ID for notifications.

    Returns:
        None.
    """
    if not offers:
        logger.info("No apartment listings to share on Telegram")
        return

    logger.info("Sending %d listings to Telegram", len(offers))

    for offer in offers:
        offer["price_point"] = get_price_point(offer, collection)

    offers.sort(key=lambda x: x.get("price_point", 0))

    for offer in offers:
        offer_text = create_offer_text(offer)
        logger.info("Sending: %s", offer_text)

        telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": offer_text}

        try:
            response = requests.post(telegram_url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to send Telegram message: %s", e)
