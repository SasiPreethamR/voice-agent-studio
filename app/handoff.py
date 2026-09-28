"""Human handoff: the built-in redirect_to_human tool, the handoff queue
(data/handoffs.json), operator WebSockets, two-way audio relay and REST routes."""

import re
import json
import uuid
import time
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from app.config import OUTPUT_SAMPLE_RATE, DATA_DIR

from app.agents import agent_manager

if TYPE_CHECKING:
    from app.voice import VoiceSession

router = APIRouter()


HANDOFFS_FILE = DATA_DIR / "handoffs.json"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


HANDOFF_TOOL_NAME = "redirect_to_human"


HANDOFF_TOOL_DESCRIPTION = (
    "Request a live human operator when the customer needs a real person, escalation, "
    "or help beyond the assistant. Include a concise summary of the customer's need "
    "so the operator can pick up the call with context."
)


HANDOFF_TOOL_FIELDS = [
    {
        "name": "Customer_Name",
        "type": "string",
        "required": True,
        "description": "Customer's name if known; otherwise use Customer",
    },
    {
        "name": "Issue_Summary",
        "type": "string",
        "required": True,
        "description": "One or two sentence summary of the customer's request, issue, and what they need from a human",
    },
    {
        "name": "Preferred_Channel",
        "type": "string",
        "required": False,
        "description": "Preferred live channel, such as voice, chat, or callback",
    },
]


def _is_handoff_tool_name(name: str) -> bool:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name or "").lower() == HANDOFF_TOOL_NAME


def get_handoff_openai_tool() -> dict:
    properties = {}
    required = []
    for field in HANDOFF_TOOL_FIELDS:
        properties[field["name"]] = {
            "type": field.get("type", "string"),
            "description": field.get("description", ""),
        }
        if field.get("required", True):
            required.append(field["name"])
    return {
        "type": "function",
        "function": {
            "name": HANDOFF_TOOL_NAME,
            "description": HANDOFF_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _tool_list_has_handoff(tools: list) -> bool:
    return any(_is_handoff_tool_name(t.get("function", {}).get("name", "")) for t in tools)


class HandoffManager:
    def __init__(self):
        self.handoffs: dict = {}
        self._load()

    def _load(self):
        if HANDOFFS_FILE.exists():
            try:
                self.handoffs = {h["id"]: h for h in json.loads(HANDOFFS_FILE.read_text())}
            except Exception:
                self.handoffs = {}

    def _save(self):
        HANDOFFS_FILE.write_text(json.dumps(list(self.handoffs.values()), indent=2))

    def create_from_tool(self, args: dict, session=None) -> dict:
        now = _utc_now()
        latest_user = ""
        transcript = []
        agent_name = ""
        if session:
            for item in session.chat_history[-12:]:
                if item.get("role") in {"user", "assistant"}:
                    transcript.append({"role": item.get("role"), "content": item.get("content", "")})
            for item in reversed(session.chat_history):
                if item.get("role") == "user":
                    latest_user = item.get("content", "")
                    break
            agent = agent_manager.get(session.active_agent_id) if session.active_agent_id else None
            agent_name = agent.get("name", "") if agent else ""
        handoff_id = "handoff_" + uuid.uuid4().hex[:12]
        handoff = {
            "id": handoff_id,
            "status": "pending",
            "customer_name": args.get("Customer_Name") or args.get("customer_name") or args.get("name") or "Customer",
            "issue_summary": args.get("Issue_Summary") or args.get("issue_summary") or args.get("query") or latest_user,
            "preferred_channel": args.get("Preferred_Channel") or args.get("preferred_channel") or "voice",
            "agent_id": session.active_agent_id if session else None,
            "agent_name": agent_name,
            "session_id": session.session_id if session else None,
            "transcript": transcript,
            "created_at": now,
            "updated_at": now,
            "accepted_at": None,
            "accepted_by": None,
            "resolved_at": None,
            "resolution_note": "",
        }
        self.handoffs[handoff_id] = handoff
        self._save()
        return handoff

    def list_all(self, status: Optional[str] = None) -> list:
        items = list(self.handoffs.values())
        if status:
            if status == "open":
                items = [h for h in items if h.get("status") in {"pending", "accepted"}]
            else:
                items = [h for h in items if h.get("status") == status]
        return sorted(items, key=lambda h: h.get("created_at", ""), reverse=True)

    def get(self, handoff_id: str) -> Optional[dict]:
        return self.handoffs.get(handoff_id)

    def update_status(self, handoff_id: str, status: str, operator_name: str = "", note: str = "") -> Optional[dict]:
        handoff = self.handoffs.get(handoff_id)
        if not handoff:
            return None
        now = _utc_now()
        handoff["status"] = status
        handoff["updated_at"] = now
        if status == "accepted":
            handoff["accepted_at"] = now
            handoff["accepted_by"] = operator_name or handoff.get("accepted_by") or "Human Agent"
        elif status == "resolved":
            handoff["resolved_at"] = now
            if note:
                handoff["resolution_note"] = note
        elif status == "cancelled" and note:
            handoff["resolution_note"] = note
        self._save()
        return handoff


handoff_manager = HandoffManager()


ACTIVE_VOICE_SESSIONS: dict[str, "VoiceSession"] = {}


HANDOFF_OPERATOR_SOCKETS: dict[str, set[WebSocket]] = {}


async def _send_audio_to_operators(handoff_id: str, audio_b64: str, sample_rate: int):
    sockets = list(HANDOFF_OPERATOR_SOCKETS.get(handoff_id, set()))
    stale = []
    for operator_ws in sockets:
        try:
            await operator_ws.send_json({
                "type": "caller_audio",
                "handoff_id": handoff_id,
                "audio": audio_b64,
                "sample_rate": sample_rate,
            })
        except Exception:
            stale.append(operator_ws)
    if stale:
        live = HANDOFF_OPERATOR_SOCKETS.get(handoff_id)
        if live:
            for operator_ws in stale:
                live.discard(operator_ws)


async def _send_audio_to_caller(handoff: dict, audio_b64: str, sample_rate: int):
    session = ACTIVE_VOICE_SESSIONS.get(handoff.get("session_id"))
    if session and session._ws and handoff.get("id") == session.active_handoff_id:
        try:
            await session._ws.send_json({
                "type": "handoff_audio",
                "handoff_id": handoff["id"],
                "audio": audio_b64,
                "sample_rate": sample_rate,
            })
        except Exception:
            pass


@router.websocket("/ws/handoffs/{handoff_id}")
async def handoff_operator_websocket(ws: WebSocket, handoff_id: str, operator_name: str = ""):
    await ws.accept()
    handoff = handoff_manager.get(handoff_id)
    if not handoff:
        await ws.send_json({"type": "error", "message": "Handoff not found"})
        await ws.close(code=1008)
        return
    HANDOFF_OPERATOR_SOCKETS.setdefault(handoff_id, set()).add(ws)
    await ws.send_json({"type": "handoff_status", "handoff": handoff})
    try:
        while True:
            data = await ws.receive()
            if "text" not in data or not data["text"]:
                continue
            msg = json.loads(data["text"])
            action = msg.get("action")
            handoff = handoff_manager.get(handoff_id)
            if not handoff:
                await ws.send_json({"type": "error", "message": "Handoff not found"})
                continue
            if action == "accept":
                handoff = handoff_manager.update_status(handoff_id, "accepted", operator_name=operator_name or msg.get("operator_name") or "Human Agent")
                if handoff:
                    await _notify_handoff_session(handoff)
                    await ws.send_json({"type": "handoff_status", "handoff": handoff})
            elif action == "operator_audio" and handoff.get("status") == "accepted":
                await _send_audio_to_caller(handoff, msg.get("audio", ""), int(msg.get("sample_rate") or OUTPUT_SAMPLE_RATE))
            elif action == "resolve":
                handoff = handoff_manager.update_status(handoff_id, "resolved", note=msg.get("note", ""))
                if handoff:
                    await _notify_handoff_session(handoff)
                    await ws.send_json({"type": "handoff_status", "handoff": handoff})
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect message" not in str(e):
            raise
    finally:
        live = HANDOFF_OPERATOR_SOCKETS.get(handoff_id)
        if live:
            live.discard(ws)
            if not live:
                HANDOFF_OPERATOR_SOCKETS.pop(handoff_id, None)


async def _notify_handoff_session(handoff: dict):
    session_id = handoff.get("session_id")
    session = ACTIVE_VOICE_SESSIONS.get(session_id) if session_id else None
    status = handoff.get("status")
    if session:
        if status == "accepted":
            session.state = "human_handoff"
            session.handoff_connected = True
            session.active_handoff_id = handoff.get("id")
        elif status in {"resolved", "cancelled"} and session.active_handoff_id == handoff.get("id"):
            session.state = "listening" if session.active else "idle"
            session.handoff_connected = False
            session.active_handoff_id = None
    if session and session._ws:
        try:
            await session._ws.send_json({"type": "handoff_status", "handoff": handoff})
        except Exception:
            pass
    stale = []
    for operator_ws in list(HANDOFF_OPERATOR_SOCKETS.get(handoff.get("id"), set())):
        try:
            await operator_ws.send_json({"type": "handoff_status", "handoff": handoff})
        except Exception:
            stale.append(operator_ws)
    live = HANDOFF_OPERATOR_SOCKETS.get(handoff.get("id"))
    if live:
        for operator_ws in stale:
            live.discard(operator_ws)


@router.get("/api/handoffs")
async def list_handoffs(status: Optional[str] = None):
    return JSONResponse({"handoffs": handoff_manager.list_all(status)})


@router.get("/api/handoffs/{handoff_id}")
async def get_handoff(handoff_id: str):
    handoff = handoff_manager.get(handoff_id)
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    return JSONResponse({"handoff": handoff})


@router.post("/api/handoffs/{handoff_id}/accept")
async def accept_handoff(handoff_id: str, data: Optional[dict] = None):
    operator_name = (data or {}).get("operator_name") or "Human Agent"
    handoff = handoff_manager.update_status(handoff_id, "accepted", operator_name=operator_name)
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})


@router.post("/api/handoffs/{handoff_id}/resolve")
async def resolve_handoff(handoff_id: str, data: Optional[dict] = None):
    handoff = handoff_manager.update_status(handoff_id, "resolved", note=(data or {}).get("note", ""))
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})


@router.post("/api/handoffs/{handoff_id}/cancel")
async def cancel_handoff(handoff_id: str, data: Optional[dict] = None):
    handoff = handoff_manager.update_status(handoff_id, "cancelled", note=(data or {}).get("note", ""))
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})
