"""
VanaciRetain Discord Bot
────────────────────────
Listens for messages in Discord and forwards them to the
FastAPI backend for processing.

Setup:
    1. Create a bot at https://discord.com/developers/applications
    2. Copy the bot token to .env as DISCORD_BOT_TOKEN
    3. Invite the bot to your server using the OAuth2 URL
    4. Start FastAPI: uvicorn api.main:app --port 8000
    5. Start the bot: uv run python -m integrations.discord_bot

The bot uses each Discord thread/channel as a conversation
thread_id, so conversation memory is preserved per channel.
"""

import os

import discord
import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "http://localhost:8000"
)
API_CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"

if not DISCORD_BOT_TOKEN:
    raise ValueError(
        "DISCORD_BOT_TOKEN not found in .env. "
        "Create a bot at https://discord.com/developers/applications"
    )


# ── Bot setup ──
intents = discord.Intents.default()
intents.message_content = True  # Required to read message text

client = discord.Client(intents=intents)
http_client = httpx.AsyncClient(timeout=30.0)


@client.event
async def on_ready():
    """Called when the bot connects to Discord."""
    print(f"\n{'='*50}")
    print(f"  VanaciRetain Discord Bot")
    print(f"  Logged in as: {client.user}")
    print(f"  API endpoint: {API_CHAT_ENDPOINT}")
    print(f"{'='*50}\n")


@client.event
async def on_message(message: discord.Message):
    """Called when any message is sent in a channel the bot can see."""

    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Respond to DMs or when mentioned
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions

    if not is_dm and not is_mentioned:
        return

    # Clean the message (remove the bot mention)
    question = message.content
    if is_mentioned:
        question = question.replace(
            f"<@{client.user.id}>", ""
        ).strip()

    if not question:
        await message.reply(
            "Please include a question after mentioning me!"
        )
        return

    print(f"[DISCORD] Question from {message.author}: {question[:60]}")

    # Use channel ID as thread_id for conversation memory
    # This means each Discord channel has its own conversation
    thread_id = f"discord_{message.channel.id}"

    # Show typing indicator while processing
    async with message.channel.typing():
        try:
            # Call the FastAPI backend
            response = await http_client.post(
                API_CHAT_ENDPOINT,
                json={
                    "question": question,
                    "thread_id": thread_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                # Format the reply
                reply = answer

                # Add sources if available
                if sources and len(answer) > 100:
                    # Clean source names (remove section info for readability)
                    clean_sources = []
                    for s in sources[:3]:
                        # "Vacation Policy (Information & Office Security)" → "Vacation Policy"
                        name = s.split("(")[0].strip()
                        if name and name not in clean_sources:
                            clean_sources.append(name)

                    if clean_sources:
                        reply += f"\n\n*{', '.join(clean_sources)}*"

                if len(reply) > 2000:
                    reply = reply[:1997] + "..."

                await message.reply(reply)

            else:
                error = response.json().get("detail", "Unknown error")
                await message.reply(
                    f"Sorry, I encountered an error: {error}"
                )

        except httpx.TimeoutException:
            await message.reply(
                "Sorry, the request timed out. "
                "The HR system might be busy. Please try again."
            )

        except httpx.ConnectError:
            await message.reply(
                "Sorry, I can't reach the HR system. "
                "Make sure the API server is running."
            )

        except Exception as e:
            print(f"[DISCORD] Error: {type(e).__name__}: {e}")
            await message.reply(
                "Sorry, something went wrong. Please try again."
            )

    print(f"[DISCORD] Replied to {message.author}")


def main():
    """Start the Discord bot."""
    print("[DISCORD] Starting bot...")
    client.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()