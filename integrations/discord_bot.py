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
def get_bot_token() -> str:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError(
            "DISCORD_BOT_TOKEN not found. "
            "Set it in .env or AWS Secrets Manager."
        )
    return token

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
    is_bot_thread = (
        isinstance(message.channel, discord.Thread)
        and message.channel.name.startswith("HR Chat -")
    )

    if not is_dm and not is_mentioned and not is_bot_thread:
        return

    # Clean the message (remove the bot mention if present)
    question = message.content
    if is_mentioned:
        question = question.replace(
            f"<@{client.user.id}>", ""
        ).strip()

    if not question:
        await message.reply("Please include a question!")
        return

    print(f"[DISCORD] Question from {message.author}: {question[:60]}")

    # Use the user's ID as thread_id for conversation memory
    thread_id = f"discord_dm_{message.author.id}"

    # If mentioned in a channel, redirect to DM for privacy
    if not is_dm:
        try:
            await message.reply(
                "I've sent you a private message with the answer! "
                "Check your DMs. All future conversations happen "
                "there for privacy."
            )
        except Exception:
            pass

        # Send the actual response via DM
        dm_channel = await message.author.create_dm()

        async with dm_channel.typing():
            try:
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

                    reply = answer
                    if sources and len(answer) > 100:
                        clean_sources = []
                        for s in sources[:3]:
                            name = s.split("(")[0].strip()
                            if name and name not in clean_sources:
                                clean_sources.append(name)
                        if clean_sources:
                            reply += f"\n\n📋 *{', '.join(clean_sources)}*"

                    if len(reply) > 2000:
                        reply = reply[:1997] + "..."

                    await dm_channel.send(reply)
                else:
                    error = response.json().get("detail", "Unknown error")
                    await dm_channel.send(f"Sorry, I encountered an error: {error}")

            except httpx.TimeoutException:
                await dm_channel.send("Sorry, the request timed out. Please try again.")
            except httpx.ConnectError:
                await dm_channel.send("Sorry, I can't reach the HR system.")
            except Exception as e:
                print(f"[DISCORD] Error: {type(e).__name__}: {e}")
                await dm_channel.send("Sorry, something went wrong. Please try again.")

        print(f"[DISCORD] Replied to {message.author} via DM")
        return

    # Already in a DM — respond directly
    async with message.channel.typing():
        try:
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

                reply = answer
                if sources and len(answer) > 100:
                    clean_sources = []
                    for s in sources[:3]:
                        name = s.split("(")[0].strip()
                        if name and name not in clean_sources:
                            clean_sources.append(name)
                    if clean_sources:
                        reply += f"\n\n📋 *{', '.join(clean_sources)}*"

                if len(reply) > 2000:
                    reply = reply[:1997] + "..."

                await message.reply(reply)
            else:
                error = response.json().get("detail", "Unknown error")
                await message.reply(f"Sorry, I encountered an error: {error}")

        except httpx.TimeoutException:
            await message.reply("Sorry, the request timed out. Please try again.")
        except httpx.ConnectError:
            await message.reply("Sorry, I can't reach the HR system.")
        except Exception as e:
            print(f"[DISCORD] Error: {type(e).__name__}: {e}")
            await message.reply("Sorry, something went wrong. Please try again.")

    print(f"[DISCORD] Replied to {message.author} via DM")


def main():
    print("[DISCORD] Starting bot")
    token = get_bot_token()
    client.run(token)

if __name__ == "__main__":
    main()