"""
SpecFlow — Slack Socket Mode bot.
Runs as a daemon thread alongside the Streamlit app.
Watches channels configured in the DB and adds messages to the queue.
No public URL needed — uses WebSocket (Socket Mode).
"""

import threading
import logging
from queue_store import add_message, get_cfg

log = logging.getLogger("specflow.slack")

_state = {"thread": None, "client": None, "running": False, "error": None}


def _resolve_channel_name(web_client, channel_id: str) -> str:
    try:
        result = web_client.conversations_info(channel=channel_id)
        return result["channel"].get("name", channel_id)
    except Exception:
        return channel_id


def _run(bot_token: str, app_token: str):
    try:
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk import WebClient
        from slack_sdk.socket_mode.response import SocketModeResponse

        wc = WebClient(token=bot_token)
        sc = SocketModeClient(app_token=app_token, web_client=wc)
        _state["client"] = sc

        def handle(client, req):
            if req.type == "events_api":
                # Acknowledge immediately
                client.send_socket_mode_response(
                    SocketModeResponse(envelope_id=req.envelope_id)
                )
                event = req.payload.get("event", {})
                # Only handle user messages, ignore bot/app messages
                if event.get("type") != "message":
                    return
                if event.get("bot_id") or event.get("subtype"):
                    return

                channel_id = event.get("channel", "")
                text = event.get("text", "").strip()
                user = event.get("user", "unknown")

                if not text:
                    return

                # Re-read config live
                watched_raw = get_cfg("slack_channels", "")
                watched = [c.strip() for c in watched_raw.split(",") if c.strip()]
                if watched and channel_id not in watched:
                    # Also try matching by name
                    ch_name = _resolve_channel_name(wc, channel_id)
                    if ch_name not in watched:
                        return

                ch_name = _resolve_channel_name(wc, channel_id)
                added = add_message(
                    source="slack",
                    content=text,
                    channel=ch_name,
                    author=user,
                )
                if added:
                    log.info(f"Queued Slack message from #{ch_name}")

        sc.socket_mode_request_listeners.append(handle)
        sc.connect()
        _state["running"] = True
        _state["error"] = None

        # Keep thread alive
        import time
        while _state["running"]:
            time.sleep(1)

    except Exception as e:
        _state["running"] = False
        _state["error"] = str(e)
        log.error(f"Slack bot error: {e}")


def start(bot_token: str, app_token: str) -> bool:
    """Start the Slack Socket Mode bot. Idempotent."""
    if _state["thread"] and _state["thread"].is_alive():
        return True
    if not bot_token or not app_token:
        return False
    t = threading.Thread(
        target=_run, args=(bot_token, app_token), daemon=True, name="slack-bot"
    )
    _state["thread"] = t
    t.start()
    return True


def stop():
    _state["running"] = False
    client = _state.get("client")
    if client:
        try:
            client.close()
        except Exception:
            pass
    _state["thread"] = None


def status() -> dict:
    t = _state["thread"]
    # Thread alive = bot is running (don't AND with _state["running"] — race condition)
    thread_alive = t is not None and t.is_alive()
    return {
        "running": thread_alive,
        "ready": thread_alive and _state["running"],   # fully initialised
        "error": _state["error"],
    }


def list_channels(bot_token: str) -> tuple[list[dict], str]:
    """
    Fetch all public channels the bot can see.
    Returns (channels, error_message). error_message is "" on success.
    """
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        wc = WebClient(token=bot_token)
        result = wc.conversations_list(types="public_channel,private_channel", limit=200)
        channels = [
            {"id": c["id"], "name": c["name"]}
            for c in result.get("channels", [])
        ]
        return channels, ""
    except Exception as e:
        return [], str(e)
