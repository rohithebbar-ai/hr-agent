"""
LocalStack Practice
───────────────────
Sets up AWS services locally using LocalStack.
Same boto3 code works on real AWS by removing endpoint_url.

Usage:
    uv run python scripts/localstack_setup.py
"""

from email import message
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()


# --- config ----
# For localstack, point at localhost:4566
# For AWS, remove endpoint entirely

LOCALSTACK_URL = os.environ.get(
    "AWS_ENDPOINT_URL", "http://localhost:4566"
)

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET_NAME = "vanacihr-documents"
SECRETS_NAME = "vanacihr/api-keys"

def get_client(service: str):
    """
    Get a boto3 client pointing at LocalStack.

    In production, remove endpoint_url and boto3 automatically
    uses real AWS credentials from IAM roles or env vars.
    """
    return boto3.client(
        service,
        endpoint_url=LOCALSTACK_URL,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

def setup_s3():
    """ Create s3 bucket and upload HR documents"""
    s3 = get_client("s3")

    # Create a bucket
    try:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"[S3] Created Bucket: {BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyExists:
        print(f"[s3] Bucket already exists: {BUCKET_NAME}")

    # Upload the raw handbook pdf
    pdf_path = Path("data/hr_documents/raw/gallagher_employee_handbook.pdf")
    if pdf_path.exists():
        s3.upload_file(
            str(pdf_path),
            BUCKET_NAME,
            "raw/gallagher_employee_handbook.pdf"
        )
        print(f"[s3] Uploaded: raw/gallagher_employee_handbook.pdf")

    # Upload processed policies.json
    policies_path = Path("data/hr_documents/processed/policies.json")
    if policies_path.exists():
        s3.upload_file(
            str(policies_path),
            BUCKET_NAME,
            "processed/policies.json",
        )
        print(f"[s3] Uploaded: processed/policies.json")

    # List bucket contents
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    print(f"\n[S3] Bucket contents:")
    for obj in response.get("Contents", []):
        print(f"{obj['Key']} ({obj['Size']} bytes)")

def setup_secrets_manager():
    """Store API keys in secrets manager instead of .env file"""
    sm = get_client("secretsmanager")

    secrets = {
        "GROQ_API_KEY" : os.environ.get("GROQ_API_KEY", ""),
        "GROQ_API_KEY_2" : os.environ.get("GROQ_API_KEY", ""),
        "GROQ_API_KEY_3" : os.environ.get("GROQ_API_KEY", ""),
        "GROQ_API_KEY_4" : os.environ.get("GROQ_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        "LANGSMITH_API_KEY": os.environ.get("LANGSMITH_API_KEY", ""),
        "DISCORD_BOT_TOKEN": os.environ.get("DISCORD_BOT_TOKEN", ""),
    }

    try:
        sm.create_secret(
            Name = SECRETS_NAME,
            SecretString = json.dumps(secrets),
        )
        print(f"[SECRETS] created: {SECRETS_NAME}")
    except sm.exceptions.ResourceExistsException:
        sm.update_secret(
            SecretId = SECRETS_NAME,
            SecretString=json.dumps(secrets)
        )
        print(f"[SECRETS] created: {SECRETS_NAME}")

    # Verify the retrievel
    response = sm.describe_secret(SecretId=SECRETS_NAME)
    print("Name:", response.get("Name"))
    print("ARN:", response.get("ARN"))
    print("Created:", response.get("CreatedDate"))
    print("Rotation enabled:", response.get("RotationEnabled"))
    print("Tags:", response.get("Tags", []))
    response_get = sm.get_secret_value(SecretId=SECRETS_NAME)
    retrieved = json.loads(response_get["SecretString"])
    print(f"[SECRETS] stored {len(retrieved)} keys")
    for key in retrieved:
        # Show key names but mask values
        val = retrieved[key]
        masked = val[:8] + "...." if len(val) > 8 else "***"
        print(f"{key}: {masked}")

def setup_sqs():
    """
    Create SQS queue for future async document processing.
    When admin uploads a PDF, a message goes here.
    A worker picks it up and runs preprocessing + ingestion.
    """
    sqs = get_client("sqs")

    try:
        response = sqs.create_queue(
            QueueName="vanaciretain-document-ingestion",
            Attributes={
                "VisibilityTimeout": "300",  # 5 min processing time
                "MessageRetentionPeriod": "86400",  # 24 hours
            },
        )
        queue_url = response["QueueUrl"]
        print(f"[SQS] Created queue: {queue_url}")

        # Send a test message
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({
                "action": "ingest_document",
                "tenant_id": "vanaciprime",
                "document_key": "raw/gallagher_employee_handbook.pdf",
                "bucket": BUCKET_NAME,
            }),
        )
        print(f"[SQS] Sent test message to queue")

        # Receive and display the message
        messages = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
        )

        if "Messages" in messages:
            msg = messages["Messages"][0]
            body = json.loads(msg["Body"])
            print(f"[SQS] Received: {body}")

            # Delete after processing
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=msg["ReceiptHandle"],
            )
            print(f"[SQS] Message processed and deleted")

    except Exception as e:
        print(f"[SQS] Error: {e}")

def main():
    print("\n" + "=" * 60)
    print("  LocalStack Practice — AWS Services Locally")
    print("=" * 60 + "\n")

    print("─" * 40)
    print("  S3 — Document Storage")
    print("─" * 40)
    setup_s3()

    print("─" * 40)
    print("secrets manager")
    setup_secrets_manager()

    print("─" * 40)
    print("SQS- message queue service")
    print("─" * 40)
    setup_sqs()

if __name__ == "__main__":
    main()