"""
AWS Secrets Helper
──────────────────
Load API keys from AWS Secrets Manager into os.environ.

- LocalStack: reads from localhost:4566 (for local dev)
- Real AWS: reads from real Secrets Manager (for EC2)

The EC2 instance's IAM role provides credentials automatically.
No access keys needed in code or .env.
"""

import json
import os
import boto3


def load_secrets_to_env(
    secret_name: str = "hragent/api-keys",
):
    """Load secrets from AWS Secrets Manager into os.environ."""

    # Auto-detect: if AWS_ENDPOINT_URL is set, use LocalStack
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    region = os.environ.get("AWS_REGION", "us-east-1")

    client_kwargs = {
        "service_name": "secretsmanager",
        "region_name": region,
    }

    if endpoint_url:
        # LocalStack mode
        client_kwargs["endpoint_url"] = endpoint_url
        client_kwargs["aws_access_key_id"] = "test"
        client_kwargs["aws_secret_access_key"] = "test"
        print(f"[SECRETS] Using LocalStack: {endpoint_url}")
    else:
        # Real AWS — boto3 auto-discovers credentials from IAM role
        print("[SECRETS] Using real AWS Secrets Manager")

    try:
        client = boto3.client(**client_kwargs)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])

        loaded = 0
        for key, value in secrets.items():
            if value:
                os.environ[key] = value
                loaded += 1

        print(f"[SECRETS] Loaded {loaded} keys successfully")

    except Exception as e:
        print(f"[SECRETS] Failed to load from Secrets Manager: {e}")
        print("[SECRETS] Falling back to environment variables / .env file")


if __name__ == "__main__":
    load_secrets_to_env()
    # Verify
    for key in ["GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(key, "")
        masked = val[:8] + "..." if len(val) > 8 else "NOT SET"
        print(f"  {key}: {masked}")