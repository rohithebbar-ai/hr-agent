import os
import json
from functools import lru_cache

import boto3


@lru_cache(maxsize=1)
def get_secrets() -> dict:
    """
    Fetch secrets once per process and cache them.
    """
    secret_name = os.getenv("AWS_SECRET_NAME", "hragent/api-keys")
    region = os.getenv("AWS_REGION", "ap-south-1")

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)

    return json.loads(response["SecretString"])


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Resolution order:
    1) Environment variable
    2) AWS Secrets Manager
    3) Default
    """
    val = os.getenv(key)
    if val:
        return val

    secrets = get_secrets()
    return secrets.get(key, default)