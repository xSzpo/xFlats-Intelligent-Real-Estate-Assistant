"""Telegram notification delivery."""

import logging
import re
from typing import Any

import chromadb
import requests

from xflats.storage.chromadb import get_price_point

logger = logging.getLogger(__name__)


def create_offer_text(
    offer_dict: dict[str, Any],
    currency: str,
) -> str:
    """Create formatted text for a single apartment offer.

    Args:
        offer_dict: Dictionary containing offer details such as address,
            price, area, and public transport information.
        currency: Currency code for price display (e.g. "PLN", "DKK").

    Returns:
        Formatted string with offer details ready for Telegram delivery.
    """
    subways_txt = ""
    if offer_dict.get("subways") and offer_dict.get("public_transport_text"):
        pattern = re.compile(r"(?<=subways:)([^;]+)(?=;)")
        subways = pattern.findall(offer_dict["public_transport_text"])
        subways_txt = ", ".join(subways)

    rent = offer_dict.get("rent")
    rent_str = f", Rent: {rent:,} {currency}" if rent else ""

    floor = offer_dict.get("floor")
    floor_str = f", Floor: {floor}" if floor else ""

    subway_line = f"\nSubway(s): {subways_txt}" if subways_txt else ""

    price = offer_dict.get("price", 0) or 0
    price_point = offer_dict.get("price_point", 0) or 0

    offer_txt = (
        f"\nAddress: {offer_dict.get('address', 'N/A')}"
        f"\nSize: {offer_dict.get('area_m2', 'N/A')} m2"
        f", Rooms: {offer_dict.get('number_of_rooms', 'N/A')}"
        f", Year: {offer_dict.get('year_built', 'N/A')}"
        f"{floor_str}{rent_str}"
        f"\nPrice: {price:,} {currency} ({price_point:.2%})"
        f"{subway_line}"
        f"\nDescription: {offer_dict.get('description', 'N/A')}"
        f"\nUrl: {offer_dict.get('url', 'N/A')}\n"
    )
    return offer_txt


def send_telegram_notifications(
    offers: list[dict[str, Any]],
    collection: chromadb.Collection,
    telegram_token: str,
    telegram_chat_id: str,
    currency: str,
) -> None:
    """Send offer notifications via Telegram.

    Args:
        offers: List of offer dictionaries containing apartment listing data.
        collection: ChromaDB collection used for price point lookups.
        telegram_token: Telegram Bot API token.
        telegram_chat_id: Target Telegram chat ID for notifications.
        currency: Currency code for price formatting.
    """
    if not offers:
        logger.info("No apartment listings to share on Telegram")
        return

    logger.info("Sending %d listings to Telegram", len(offers))

    for offer in offers:
        offer["price_point"] = get_price_point(offer, collection)

    offers.sort(key=lambda x: x.get("price_point", 0))

    for offer in offers:
        offer_text = create_offer_text(offer, currency=currency)
        logger.info("Sending: %s", offer_text)

        telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": offer_text}

        try:
            response = requests.post(telegram_url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to send Telegram message: %s", e)
