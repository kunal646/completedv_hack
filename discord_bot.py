"""
SpecFlow — Discord WebSocket bot.
Runs as a daemon thread alongside the Streamlit app.
Watches channels configured in the DB and adds messages to the queue.
"""

import asyncio
import threading
import logging
from queue_store import add_message, get_cfg

log = logging.getLogger("specflow.discord")

# Module-level singleton — survives Streamlit reruns
_state = {"thread": None, "client": None, "running": False, "error": None}


def _run(token: str):
    try:
        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        _state["client"] = client

        @client.event
        async def on_ready():
            log.info(f"Discord bot ready as {client.user}")
            _state["running"] = True
            _state["error"] = None

        @client.event
        async def on_message(message: discord.Message):
            if message.author.bot:
                return
            # Re-read config each message so channel updates are live
            watched_raw = get_cfg("discord_channels", "")
            watched = [c.strip() for c in watched_raw.split(",") if c.strip()]
            # Match by channel ID or channel name
            ch_id   = str(message.channel.id)
            ch_name = str(message.channel.name)
            if watched and ch_id not in watched and ch_name not in watched:
                return
            added = add_message(
                source="discord",
                content=message.content,
                channel=ch_name,
                author=str(message.author.display_name),
            )
            if added:
                log.info(f"Queued Discord message from #{ch_name}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(client.start(token))

    except Exception as e:
        _state["running"] = False
        _state["error"] = str(e)
        log.error(f"Discord bot error: {e}")


def start(token: str) -> bool:
    """Start the Discord bot in a background thread. Idempotent."""
    if _state["thread"] and _state["thread"].is_alive():
        return True  # already running
    if not token:
        return False
    t = threading.Thread(target=_run, args=(token,), daemon=True, name="discord-bot")
    _state["thread"] = t
    t.start()
    return True


def stop():
    client = _state.get("client")
    if client:
        asyncio.run_coroutine_threadsafe(client.close(), client.loop)
    _state["running"] = False
    _state["thread"] = None


def status() -> dict:
    t = _state["thread"]
    return {
        "running": t is not None and t.is_alive() and _state["running"],
        "error": _state["error"],
    }
