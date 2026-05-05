"""
VanaciRetain Discord Bot (Production Ready)
──────────────────────────────────────────
- Uses AWS Secrets Manager directly (no aws_secrets.py)
- Cached secrets (no repeated AWS calls)
- Docker-compatible (service networking)
"""

import json
import os
from functools import lru_cache

import discord
import httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# AWS Secrets Loader (centralized + cached)
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_secrets() -> dict:
    import boto3

    secret_name = os.getenv("AWS_SECRET_NAME", "hragent/api-keys")
    region = os.getenv("AWS_REGION", "ap-south-1")

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)

    return json.loads(response["SecretString"])


def get_secret(key: str, default: str | None = None) -> str | None:
    # Priority 1: environment
    value = os.getenv(key)
    if value:
        return value

    # Priority 2: AWS Secrets Manager
    try:
        secrets = _get_secrets()
        return secrets.get(key, default)
    except Exception as e:
        print(f"[SECRETS] Error fetching {key}: {e}")
        return default


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
def get_bot_token() -> str:
    token = get_secret("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not found")
    return token


API_BASE_URL = get_secret("API_BASE_URL", "http://app:8000")
API_CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"


# ─────────────────────────────────────────────
# Bot setup
# ─────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
http_client = httpx.AsyncClient(timeout=30.0)


@client.event
async def on_ready():
    print("\n" + "=" * 50)
    print("VanaciRetain Discord Bot")
    print(f"Logged in as: {client.user}")
    print(f"API endpoint: {API_CHAT_ENDPOINT}")
    print("=" * 50 + "\n")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions

    if not is_dm and not is_mentioned:
        return

    question = message.content.replace(f"<@{client.user.id}>", "").strip()

    if not question:
        await message.reply("Please include a question!")
        return

    print(f"[DISCORD] {message.author}: {question[:80]}")

    thread_id = f"discord_{message.author.id}"

    async with message.channel.typing():
        try:
            response = await http_client.post(
                API_CHAT_ENDPOINT,
                json={"question": question, "thread_id": thread_id},
            )

            if response.status_code == 200:
                data = response.json()
                reply = data["answer"]

                if len(reply) > 2000:
                    reply = reply[:1997] + "..."

                await message.reply(reply)

            else:
                await message.reply("Error processing request")

        except httpx.ConnectError:
            await message.reply("Cannot reach backend service")
        except Exception as e:
            print(f"[ERROR] {e}")
            await message.reply("Something went wrong")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    print("[DISCORD] Starting bot")
    token = get_bot_token()
    client.run(token)


if __name__ == "__main__":
    main()