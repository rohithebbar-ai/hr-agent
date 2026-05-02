"""
AWS Secrets Helper
──────────────────
Load API keys from AWS Secrets Manager instead of .env files.

In development (LocalStack): reads from localhost:4566
In production (real AWS): reads from real Secrets Manager

Usage:
    from scripts.aws_secrets import load_secrets_to_env
    load_secrets_to_env()  # Sets os.environ with all keys
"""

import json
import os

import boto3


def load_secrets_to_env(
    secret_name: str = "vanacihr/api-keys",
    use_localstack: bool = None,
):
    """
    Load secrets from AWS Secrets Manager into os.environ.

    Args:
        secret_name: The secret ID in Secrets Manager
        use_localstack: If True, use LocalStack. If None,
                       auto-detect from AWS_ENDPOINT_URL env var.
    """
    if use_localstack is None:
        use_localstack = bool(os.environ.get("AWS_ENDPOINT_URL"))

    client_kwargs = {
        "service_name": "secretsmanager",
        "region_name": os.environ.get("AWS_REGION", "us-east-1"),
    }

    if use_localstack:
        endpoint = os.environ.get(
            "AWS_ENDPOINT_URL", "http://localhost:4566"
        )
        client_kwargs["endpoint_url"] = endpoint
        client_kwargs["aws_access_key_id"] = "test"
        client_kwargs["aws_secret_access_key"] = "test"

    try:
        client = boto3.client(**client_kwargs)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])

        for key, value in secrets.items():
            if value:
                os.environ[key] = value

        print(
            f"[SECRETS] Loaded {len(secrets)} keys "
            f"from {'LocalStack' if use_localstack else 'AWS'}"
        )

    except Exception as e:
        print(f"[SECRETS] Failed to load: {e}")
        print("[SECRETS] Falling back to .env file")