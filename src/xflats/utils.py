"""Shared utility functions."""

from __future__ import annotations

import time

import requests


def geocode_address(address: str) -> tuple[float, float]:
    """Use OSM Nominatim to turn a street address into (lat, lon)."""
    url = "https://nominatim.openstreetmap.org/search"
    params: dict[str, str | int] = {"q": address, "format": "json", "limit": 1}
    headers = {
        "User-Agent": "xflats/1.0 (github.com/xSzpo/xFlats-Intelligent-Real-Estate-Assistant)"
    }
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No location found for address: {address!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def get_public_transport_stations(
    lat: float | None = None,
    lon: float | None = None,
    address: str | None = None,
    radius: int = 700,
    max_retries: int = 5,
) -> dict[str, bool | str]:
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

    return {
        "ferry_terminals": False,
        "light_rails": False,
        "subways": False,
        "bus_stations": False,
        "trains": False,
        "public_transport_text": "",
    }


def remove_url_parameters(url: str) -> str:
    return url.split("?", 1)[0]


def filter_unique_ids(dict_list: list[dict], id_key: str = "id") -> list[dict]:
    seen_ids = set()
    result = []
    for item in dict_list:
        if item.get(id_key):
            unique_id = item.get(id_key)
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                result.append(item)
    return result
