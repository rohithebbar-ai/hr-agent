import os
import json
from functools import lru_cache

import boto3
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_secrets() -> dict:
    """
    Fetch secrets from AWS Secrets Manager.
    Returns empty dict if AWS is not reachable (local dev).
    """
    secret_name = os.getenv("AWS_SECRET_NAME", "hragent/api-keys")
    region = os.getenv("AWS_REGION", "ap-south-1")

    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])
        print(f"[SECRETS] Loaded {len(secrets)} keys from AWS Secrets Manager")
        return secrets
    except Exception as e:
        print(f"[SECRETS] AWS not available ({type(e).__name__}), using .env only")
        return {}


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Resolution order:
    1) Environment variable (.env or shell)
    2) AWS Secrets Manager
    3) Default
    """
    val = os.getenv(key)
    if val:
        return val

    secrets = get_secrets()
    return secrets.get(key, default)