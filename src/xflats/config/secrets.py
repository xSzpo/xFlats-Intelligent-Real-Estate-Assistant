"""Secret management and application configuration."""

import json
import os

import boto3

DEFAULT_PAGES_TO_OPEN = 3
DEFAULT_NUMBER_OF_ROOMS = 2


def get_secret(secret_id, key=None, profile_name=None):
    session = boto3.session.Session(profile_name=profile_name)
    client = session.client(service_name="secretsmanager", region_name="eu-central-1")
    get_secret_value_response = client.get_secret_value(SecretId=secret_id)
    secret = json.loads(get_secret_value_response["SecretString"])
    if key:
        if key not in secret:
            raise KeyError(f"Key {key!r} not found in secret {secret_id!r}. Available keys: {list(secret.keys())}")
        return secret[key]
    return secret


class Config:
    def __init__(self):
        self.profile_name = os.getenv("AWS_PROFILE", None)
        self.chromadb_ip = os.getenv("CHROMADB_IP", "chromadb")
        self.number_of_pages_to_open = int(os.getenv("NUMBER_OF_PAGES_TO_OPEN", DEFAULT_PAGES_TO_OPEN))
        self.number_of_rooms = int(os.getenv("NUMBER_OF_ROOMS", DEFAULT_NUMBER_OF_ROOMS))
        self.telegram_token = get_secret(secret_id="telegram-274181059559", key="TOKEN", profile_name=self.profile_name)
        self.telegram_chat_id = get_secret(secret_id="telegram-274181059559", key="CHAT_ID", profile_name=self.profile_name)
        self.genai_api_key = get_secret(secret_id="gemini-274181059559", key="GOOGLE_API_KEY", profile_name=self.profile_name)
