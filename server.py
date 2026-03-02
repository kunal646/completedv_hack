"""
SpecFlow — FastAPI backend
Serves the frontend + all REST API endpoints.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from queue_store import (
    add_message, get_messages, set_status, board_counts,
    get_items, get_item, update_item, delete_item,
    save_auto_process_result, save_tech_plan,
    get_cfg, set_cfg, LANE_META, LANE_STATUSES,
)
from agent import auto_process, build_architect_graph
import discord_bot
import slack_bot

app = FastAPI(title="SpecFlow API")
executor = ThreadPoolExecutor(max_workers=4)
_processing: set[int] = set()   # item IDs currently being LLM-processed

STATIC_DIR = Path(__file__).parent / "static"


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    dt = get_cfg("discord_token")
    if dt:
        discord_bot.start(dt)
    sbt = get_cfg("slack_bot_token")
    sat = get_cfg("slack_app_token")
    if sbt and sat:
        slack_bot.start(sbt, sat)


# ── Static files + SPA fallback ────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── API: Counts ────────────────────────────────────────────────────────────────

@app.get("/api/counts")
def get_counts():
    c = board_counts()
    return {
        "inbox": c.get("pending", 0),
        "lanes": {lane: c.get(lane, 0) for lane in ["issue", "feature", "idea", "marketing"]},
        "processing": list(_processing),
    }


# ── API: Inbox ─────────────────────────────────────────────────────────────────

@app.get("/api/inbox")
def get_inbox():
    msgs = get_messages("pending")
    return {"items": msgs, "processing": list(_processing)}


class AddMessageBody(BaseModel):
    content: str
    source: str = "manual"
    channel: Optional[str] = "manual"
    author: Optional[str] = "You"


@app.post("/api/inbox")
def add_inbox_message(body: AddMessageBody):
    ok = add_message(body.source, body.content, body.channel, body.author)
    return {"success": ok, "message": "Duplicate" if not ok else "Added"}


@app.delete("/api/inbox/{msg_id}")
def dismiss_inbox_message(msg_id: int):
    set_status(msg_id, "dismissed")
    return {"success": True}


# ── API: Auto-Process ──────────────────────────────────────────────────────────

class ProcessBody(BaseModel):
    api_key: str


@app.post("/api/process/{msg_id}")
async def process_message(msg_id: int, body: ProcessBody):
    if msg_id in _processing:
        return {"success": False, "message": "Already processing"}

    item = get_item(msg_id)
    if not item:
        raise HTTPException(404, "Message not found")

    _processing.add(msg_id)
    loop = asyncio.get_event_loop()

    def do_process():
        try:
            result = auto_process(item["content"], body.api_key)
            save_auto_process_result(
                msg_id,
                result["lane"], result["title"],
                result["summary"], result["priority"],
                result["output"] or "", result["needs_review"],
            )
        except Exception as e:
            print(f"[SpecFlow] Processing error for {msg_id}: {e}")
            set_status(msg_id, "pending")  # return to inbox on failure
        finally:
            _processing.discard(msg_id)

    loop.run_in_executor(executor, do_process)
    return {"success": True, "message": "Processing started"}


# ── API: Board ─────────────────────────────────────────────────────────────────

@app.get("/api/board")
def get_board(lane: Optional[str] = None, status: Optional[str] = None):
    lanes = ["issue", "feature", "idea", "marketing"]
    if lane and lane in lanes:
        return {lane: get_items(lane=lane, item_status=status or None)}
    return {l: get_items(lane=l, item_status=status or None) for l in lanes}


@app.get("/api/item/{item_id}")
def get_item_detail(item_id: int):
    item = get_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item


class UpdateItemBody(BaseModel):
    lane: Optional[str] = None
    title: Optional[str] = None
    priority: Optional[str] = None
    item_status: Optional[str] = None
    notes: Optional[str] = None


@app.put("/api/item/{item_id}")
def update_item_endpoint(item_id: int, body: UpdateItemBody):
    kwargs = {k: v for k, v in body.dict().items() if v is not None}
    update_item(item_id, **kwargs)
    return {"success": True}


@app.delete("/api/item/{item_id}")
def delete_item_endpoint(item_id: int):
    delete_item(item_id)
    return {"success": True}


# ── API: Build Plan (for issues after human review) ───────────────────────────

class BuildPlanBody(BaseModel):
    api_key: str
    issue_card: str


@app.post("/api/item/{item_id}/build-plan")
async def build_plan(item_id: int, body: BuildPlanBody):
    if item_id in _processing:
        return {"success": False, "message": "Already processing"}

    item = get_item(item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    _processing.add(item_id)
    loop = asyncio.get_event_loop()

    def do_plan():
        try:
            g = build_architect_graph(body.api_key)
            r = g.invoke({
                "issue_card": body.issue_card,
                "tech_plan": None, "error": None, "debug": None,
            })
            if not r.get("error"):
                save_tech_plan(item_id, r["tech_plan"])
        except Exception as e:
            print(f"[SpecFlow] Plan error for {item_id}: {e}")
        finally:
            _processing.discard(item_id)

    loop.run_in_executor(executor, do_plan)
    return {"success": True}


# ── API: Bots ──────────────────────────────────────────────────────────────────

@app.get("/api/bots/status")
def get_bot_status():
    return {
        "discord": discord_bot.status(),
        "slack": slack_bot.status(),
    }


class DiscordConfig(BaseModel):
    token: str
    channels: str = ""


@app.post("/api/bots/discord")
def connect_discord(config: DiscordConfig):
    set_cfg("discord_token", config.token)
    set_cfg("discord_channels", config.channels)
    discord_bot.stop()
    time.sleep(0.5)
    ok = discord_bot.start(config.token)
    return {"success": ok}


@app.delete("/api/bots/discord")
def disconnect_discord():
    discord_bot.stop()
    set_cfg("discord_token", "")
    return {"success": True}


class SlackConfig(BaseModel):
    bot_token: str
    app_token: str
    channels: str = ""


@app.post("/api/bots/slack")
def connect_slack(config: SlackConfig):
    set_cfg("slack_bot_token", config.bot_token)
    set_cfg("slack_app_token", config.app_token)
    set_cfg("slack_channels", config.channels)
    slack_bot.stop()
    time.sleep(0.5)
    ok = slack_bot.start(config.bot_token, config.app_token)
    return {"success": ok}


@app.get("/api/bots/slack/channels")
def fetch_slack_channels(token: str):
    channels, err = slack_bot.list_channels(token)
    return {"channels": channels, "error": err}


@app.delete("/api/bots/slack")
def disconnect_slack():
    slack_bot.stop()
    set_cfg("slack_bot_token", "")
    return {"success": True}


# ── API: Config ────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return {
        "discord_connected": bool(get_cfg("discord_token")),
        "discord_channels": get_cfg("discord_channels"),
        "slack_connected": bool(get_cfg("slack_bot_token")),
        "slack_channels": get_cfg("slack_channels"),
    }
