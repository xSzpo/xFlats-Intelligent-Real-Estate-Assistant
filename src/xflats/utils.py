"""Shared utility functions."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def geocode_address(
    address: str,
    country: str | None = None,
    address_cleanup: bool = False,
) -> tuple[float, float]:
    """Use OSM Nominatim to turn a street address into (lat, lon).

    Args:
        address: Free-text street address to geocode.
        country: Optional country name to append for better geocoding
            accuracy (e.g. "Polska", "Denmark").
        address_cleanup: If ``True``, strip common prefixes like "ul."
            that can cause geocoding failures.

    Returns:
        A ``(latitude, longitude)`` tuple of floats.

    Raises:
        ValueError: If Nominatim returns no results for the address.
        requests.HTTPError: If the HTTP request fails.
    """
    geocode_query = address
    if address_cleanup:
        geocode_query = geocode_query.replace("ul. ", "").replace("ul.", "")
    if country and country not in geocode_query:
        geocode_query = f"{geocode_query}, {country}"

    url = "https://nominatim.openstreetmap.org/search"
    params: dict[str, str | int] = {"q": geocode_query, "format": "json", "limit": 1}
    headers = {
        "User-Agent": "xflats/1.0 (github.com/xSzpo/xFlats-Intelligent-Real-Estate-Assistant)"
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No location found for address: {geocode_query!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def get_public_transport_stations(
    lat: float | None = None,
    lon: float | None = None,
    address: str | None = None,
    radius: int = 700,
    max_retries: int = 2,
) -> dict[str, bool | str]:
    """Query Overpass API for public-transport stations near a point.

    Args:
        lat: Latitude of the search centre.
        lon: Longitude of the search centre.
        address: Street address (geocoded automatically when provided).
        radius: Search radius in metres.
        max_retries: Maximum number of retry attempts on failure.

    Returns:
        A dict with boolean flags per transport type and a
        ``public_transport_text`` summary string.
    """
    if address is not None:
        lat, lon = geocode_address(address)

    if lat is None or lon is None:
        raise ValueError("Either 'address' or both 'lat' and 'lon' must be provided.")

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      node(around:{radius},{lat},{lon})[public_transport=station];
      node(around:{radius},{lat},{lon})[railway=subway_entrance];
      node(around:{radius},{lat},{lon})[railway=station];
    );
    out body;
    """

    backoff_time = 10
    station_types = {
        "ferry_terminals": "ferry_terminal",
        "light_rails": "light_rail",
        "subways": "subway",
        "bus_stations": "bus_station",
        "trains": "train",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(overpass_url, params={"data": query}, timeout=20)
            response.raise_for_status()
            data = response.json()

            stations: dict[str, list[str]] = {key: [] for key in station_types}
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

            result: dict[str, bool | str] = {}
            public_transport_text = ""
            for key in stations:
                if stations[key]:
                    public_transport_text += (
                        "; ".join(f"{key}:{i}" for i in set(stations[key])) + "; "
                    )
                    result[key] = True
                else:
                    result[key] = False

            result["public_transport_text"] = public_transport_text
            return result

        except requests.exceptions.RequestException as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                logger.warning("Failed after %d attempts: %s", max_retries, e)
                return {
                    "ferry_terminals": False,
                    "light_rails": False,
                    "subways": False,
                    "bus_stations": False,
                    "trains": False,
                    "public_transport_text": "",
                }

    return {
        "ferry_terminals": False,
        "light_rails": False,
        "subways": False,
        "bus_stations": False,
        "trains": False,
        "public_transport_text": "",
    }


def remove_url_parameters(url: str) -> str:
    """Strip query parameters from a URL.

    Args:
        url: Full URL string.

    Returns:
        The URL without any query-string portion.
    """
    return url.split("?", 1)[0]


def filter_unique_ids(
    dict_list: list[dict[str, Any]], id_key: str = "id"
) -> list[dict[str, Any]]:
    """De-duplicate a list of dicts by a given key.

    Args:
        dict_list: Input list of dictionaries.
        id_key: Key whose value is used as the unique identifier.

    Returns:
        A new list containing only the first occurrence of each unique id.
    """
    seen_ids = set()
    result = []
    for item in dict_list:
        if item.get(id_key):
            unique_id = item.get(id_key)
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                result.append(item)
    return result


def filter_none_metadata(
    offers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter out None values from offer metadata dicts.

    ChromaDB only accepts str, int, float, and bool metadata values.
    This strips any keys with ``None`` values.

    Args:
        offers: List of offer dictionaries.

    Returns:
        New list of dicts with ``None``-valued keys removed.
    """
    return [{k: v for k, v in offer.items() if v is not None} for offer in offers]
