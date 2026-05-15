"""Secret management and application configuration."""

import json
import os
from typing import Any

import boto3

DEFAULT_PAGES_TO_OPEN = 3
DEFAULT_NUMBER_OF_ROOMS = 2

_clients: dict[str | None, Any] = {}


def _get_client(profile_name: str | None = None) -> Any:
    """Get or create a cached AWS Secrets Manager client.

    Args:
        profile_name: AWS profile name to use for the session.
            Defaults to None (uses default credentials).

    Returns:
        A boto3 Secrets Manager client for the given profile.
    """
    if profile_name not in _clients:
        session = boto3.session.Session(profile_name=profile_name)
        _clients[profile_name] = session.client(
            service_name="secretsmanager", region_name="eu-central-1"
        )
    return _clients[profile_name]


def get_secret(
    secret_id: str, key: str | None = None, profile_name: str | None = None
) -> str:
    """Retrieve a secret value from AWS Secrets Manager.

    Args:
        secret_id: The ID or ARN of the secret to retrieve.
        key: Optional key within the JSON secret to return. If None,
            returns the entire parsed secret dict.
        profile_name: AWS profile name for the session.

    Returns:
        The secret value for the given key, or the full parsed secret dict.

    Raises:
        KeyError: If the specified key is not found in the secret.
    """
    client = _get_client(profile_name)
    get_secret_value_response = client.get_secret_value(SecretId=secret_id)
    secret = json.loads(get_secret_value_response["SecretString"])
    if key:
        if key not in secret:
            raise KeyError(
                f"Key {key!r} not found in secret {secret_id!r}. Available keys: {list(secret.keys())}"
            )
        return secret[key]
    return secret


class Config:
    """Application configuration loaded from environment and AWS Secrets Manager.

    Reads environment variables for infrastructure settings and fetches
    sensitive credentials (Telegram, Gemini API keys) from Secrets Manager.
    """

    def __init__(self) -> None:
        """Initialize configuration from environment variables and secrets."""
        self.profile_name = os.getenv("AWS_PROFILE", None)
        self.chromadb_ip = os.getenv("CHROMADB_IP", "chromadb")
        self.number_of_pages_to_open = int(
            os.getenv("NUMBER_OF_PAGES_TO_OPEN", DEFAULT_PAGES_TO_OPEN)
        )
        self.number_of_rooms = int(
            os.getenv("NUMBER_OF_ROOMS", DEFAULT_NUMBER_OF_ROOMS)
        )
        self.telegram_token = get_secret(
            secret_id="telegram-274181059559",
            key="TOKEN",
            profile_name=self.profile_name,
        )
        self.telegram_chat_id = get_secret(
            secret_id="telegram-274181059559",
            key="CHAT_ID",
            profile_name=self.profile_name,
        )
        self.genai_api_key = get_secret(
            secret_id="gemini-274181059559",
            key="GOOGLE_API_KEY",
            profile_name=self.profile_name,
        )
