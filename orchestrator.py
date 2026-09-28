"""
Voice Agent Orchestrator - Multi-Agent System
Port 8080 - Serves UI, REST API for RAG, WebSocket for real-time voice.

Pipeline:  Mic -> STT (8010) -> vLLM/Gemma (8003) + RAG -> TTS (8011) -> Speaker
Indic languages use the AI4Bharat server (7862) for STT/TTS. Any stage can be
switched to the Gemini API per call (BYOK) - see model_providers.py.
Each agent has its own instructions, API access, voice/language/speed settings.
A switch_agent tool is auto-injected so agents can hand off to each other.
"""

import re
import json
import uuid
import base64
import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import STT_URL, LLM_URL, TTS_URL, INDIC_URL, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
from config import GEMINI_LLM_MODEL, GEMINI_STT_MODEL, GEMINI_TTS_MODEL, GEMINI_REASONING_EFFORT, GEMINI_API_KEY
from config import MODEL_PROVIDER, GEMINI_ONLY
from model_providers import (
    LANG_NAMES, KOKORO_LANG_CODES, GEMINI_VOICES, ModelSettings, ProviderError,
    transcribe_audio, stream_llm_tokens, call_llm_with_tools, synthesize_speech_stream, gemini_list_models,
)
from rag_pipeline import RAGPipeline

# Raise Starlette's multipart limits so large folder uploads (up to 100k files) work.
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_files = 100_000
    MultiPartParser.max_fields = 100_000
except Exception:
    pass

# -- VAD constants --
ENERGY_THRESHOLD = 600
INTERRUPT_ENERGY_THRESHOLD = 700
SILENCE_TURN_END_MS = 800
CHUNK_MS = 100
MIN_SPEECH_FRAMES = 3
MIN_INTERRUPT_FRAMES = 3

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. You are having a live voice "
    "conversation. Keep your answers concise (1-3 sentences unless more detail "
    "is requested). Use natural, conversational language. Avoid markdown, lists, "
    "or formatting. If context from documents is provided, use it as the primary "
    "source for document-specific questions. If the retrieved document context is "
    "missing or insufficient, say that you do not see it in the selected documents "
    "instead of guessing."
)

# -- App --
app = FastAPI(title="Voice Agent Orchestrator", version="2.0.0")
rag = RAGPipeline()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Pre-load tool-wait audio cue (PCM, 24kHz)
TOOL_WAIT_AUDIO_B64 = ""
_twpath = STATIC_DIR / "tool_wait.pcm"
if _twpath.exists():
    TOOL_WAIT_AUDIO_B64 = base64.b64encode(_twpath.read_bytes()).decode()
    print(f"[ORCH] Loaded tool-wait audio ({len(TOOL_WAIT_AUDIO_B64)//1024}KB b64)")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TOOLS_FILE = DATA_DIR / "tools.json"
AGENTS_FILE = DATA_DIR / "agents.json"
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


# ================================================================
# Tool Library
# ================================================================
class ToolLibrary:
    def __init__(self):
        self.tools: dict = {}
        self._load()

    def _load(self):
        if TOOLS_FILE.exists():
            try:
                self.tools = {t["id"]: t for t in json.loads(TOOLS_FILE.read_text())}
            except Exception:
                self.tools = {}

    def _save(self):
        TOOLS_FILE.write_text(json.dumps(list(self.tools.values()), indent=2))

    def add(self, tool: dict) -> dict:
        tool["id"] = tool.get("id") or str(uuid.uuid4())[:8]
        self.tools[tool["id"]] = tool
        self._save()
        return tool

    def update(self, tool_id: str, data: dict) -> Optional[dict]:
        if tool_id not in self.tools:
            return None
        data["id"] = tool_id
        self.tools[tool_id] = data
        self._save()
        return data

    def delete(self, tool_id: str) -> bool:
        if tool_id in self.tools:
            del self.tools[tool_id]
            self._save()
            return True
        return False

    def get(self, tool_id: str) -> Optional[dict]:
        return self.tools.get(tool_id)

    def get_by_name(self, name: str) -> Optional[dict]:
        for t in self.tools.values():
            if t["name"] == name:
                return t
        return None

    def list_all(self) -> list:
        return list(self.tools.values())

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    def to_openai_tools(self, tool_ids: list = None) -> list:
        """Convert tool definitions to OpenAI format. If tool_ids given, filter."""
        tools = []
        for t in self.tools.values():
            if tool_ids is not None and t["id"] not in tool_ids:
                continue
            properties = {}
            required = []
            for f in t.get("fields", []):
                properties[f["name"]] = {
                    "type": f.get("type", "string"),
                    "description": f.get("description", ""),
                }
                if f.get("required", True):
                    required.append(f["name"])
            tools.append({
                "type": "function",
                "function": {
                    "name": self._sanitize_name(t["name"]),
                    "description": t["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        return tools

    def get_name_mapping(self) -> dict:
        return {self._sanitize_name(t["name"]): t["name"] for t in self.tools.values()}


# ================================================================
# Agent Manager
# ================================================================
class AgentManager:
    def __init__(self):
        self.agents: dict = {}
        self._load()

    def _load(self):
        if AGENTS_FILE.exists():
            try:
                self.agents = {a["id"]: a for a in json.loads(AGENTS_FILE.read_text())}
            except Exception:
                self.agents = {}

    def _save(self):
        AGENTS_FILE.write_text(json.dumps(list(self.agents.values()), indent=2))

    def add(self, agent: dict) -> dict:
        agent["id"] = agent.get("id") or str(uuid.uuid4())[:8]
        self.agents[agent["id"]] = agent
        self._save()
        return agent

    def update(self, agent_id: str, data: dict) -> Optional[dict]:
        if agent_id not in self.agents:
            return None
        data["id"] = agent_id
        self.agents[agent_id] = data
        self._save()
        return data

    def delete(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._save()
            return True
        return False

    def get(self, agent_id: str) -> Optional[dict]:
        return self.agents.get(agent_id)

    def list_all(self) -> list:
        return list(self.agents.values())

    def get_switch_tool(self) -> dict:
        """Returns a switch_agent tool definition for OpenAI format."""
        agent_names = [a["name"] for a in self.agents.values()]
        return {
            "type": "function",
            "function": {
                "name": "switch_agent",
                "description": (
                    "Switch the conversation to a different agent. Use this when the user's request "
                    "is better handled by another agent, or when the user asks to speak to a specific agent. "
                    "Available agents: " + ", ".join(agent_names)
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "The name of the agent to switch to. Must be one of: " + ", ".join(agent_names),
                        }
                    },
                    "required": ["agent_name"],
                }
            }
        }


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


tool_library = ToolLibrary()
agent_manager = AgentManager()
handoff_manager = HandoffManager()
ACTIVE_VOICE_SESSIONS: dict[str, "VoiceSession"] = {}
HANDOFF_OPERATOR_SOCKETS: dict[str, set[WebSocket]] = {}


# ================================================================
# Voice Session
# ================================================================
class VoiceSession:
    def __init__(self):
        self.session_id = "sess_" + uuid.uuid4().hex[:12]
        self.active = True
        self.state = "idle"
        self.audio_buffer = bytearray()
        self.pre_buffer = bytearray()
        self.is_user_speaking = False
        self.speech_frame_count = 0
        self.interrupt_frame_count = 0
        self.silence_frames = 0
        self.chat_history: list = []
        self.active_agent_id: Optional[str] = None
        self.language = "en"
        self.voice = "alloy"
        self.speed = 1.0
        self.interrupt_event = asyncio.Event()
        self.turn_complete_event = asyncio.Event()
        self.partial_text = ""
        self._llm_task: Optional[asyncio.Task] = None
        self.pending_tts_tasks: list = []
        self._ws: Optional[WebSocket] = None
        self.active_handoff_id: Optional[str] = None
        self.handoff_connected = False
        self.model = ModelSettings()

    def apply_agent(self, agent_id: str):
        """Load agent settings into this session."""
        ag = agent_manager.get(agent_id)
        if ag:
            self.active_agent_id = agent_id
            self.language = ag.get("language", "en")
            self.voice = ag.get("voice", "alloy")
            self.speed = ag.get("speed", 1.0)

    def get_agent_instructions(self) -> str:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("instructions", "")
        return ""

    def get_agent_tool_ids(self) -> list:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("tool_ids", [])
        return []

    def get_agent_document_scopes(self) -> list:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("document_scopes", [])
        return []


# ================================================================
# Audio Helpers
# ================================================================
def calc_energy(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


# ================================================================
# Service Calls
# ================================================================
def _parse_raw_tool_call(text: str) -> tuple:
    m = re.search(r'<\|tool_call>call:([^{]+)\{(.+?)\}', text)
    if not m:
        return None, None
    fn_name = m.group(1).strip()
    raw_args = m.group(2)
    args = {}
    for pair in re.finditer(r'([\w]+):<\|"\|>(.+?)<\|"\|>', raw_args):
        args[pair.group(1)] = pair.group(2)
    return fn_name, args


async def execute_tool_call(tool_name: str, args: dict, session=None) -> str:
    """Execute a tool call. Handles switch_agent specially."""
    # Handle switch_agent
    if tool_name == "switch_agent":
        target_name = args.get("agent_name", "")
        for ag in agent_manager.list_all():
            if ag["name"].lower() == target_name.lower():
                if session:
                    session.apply_agent(ag["id"])
                return json.dumps({"status": "switched", "agent_name": ag["name"], "agent_id": ag["id"]})
        return json.dumps({"error": f"Agent '{target_name}' not found"})

    # Built-in human handoff tool is always available, even if it is not selected
    # in the agent's custom API list.
    name_map = tool_library.get_name_mapping()
    original_name = name_map.get(tool_name, tool_name)
    if _is_handoff_tool_name(original_name) or _is_handoff_tool_name(tool_name):
        handoff = handoff_manager.create_from_tool(args or {}, session=session)
        if session:
            session.active_handoff_id = handoff["id"]
            session.handoff_connected = False
        payload = {
            "status": "handoff_requested",
            "handoff_id": handoff["id"],
            "customer_name": handoff["customer_name"],
            "operator_console": "/human",
            "message": "Human handoff request queued. A human operator can accept it from the console.",
        }
        if session and session._ws:
            try:
                await session._ws.send_json({"type": "handoff_status", "handoff": handoff})
            except Exception:
                pass
        return json.dumps(payload)

    # Regular tool
    tool = tool_library.get_by_name(original_name) or tool_library.get_by_name(tool_name)
    if not tool:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    if tool.get("mode") == "realtime" and tool.get("curl"):
        try:
            parsed = _parse_curl_server(tool["curl"], args)
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.request(
                    parsed["method"], parsed["url"],
                    headers=parsed.get("headers", {}),
                    json=parsed.get("body") if parsed.get("body") else None,
                )
                raw = resp.text
                # Check if response is SSE (Server-Sent Events) stream
                if raw.strip().startswith("data:"):
                    return _parse_sse_response(raw)
                return raw[:4000]
        except Exception as e:
            return json.dumps({"error": f"API call failed: {str(e)}"})
    else:
        if tool.get("sample_output"):
            return tool["sample_output"]
        return json.dumps({
            "status": "success", "tool": tool_name,
            "message": f"Mock execution of {tool_name} completed successfully.",
            "parameters_received": args,
        })


def _parse_curl_server(curl_template: str, args: dict) -> dict:
    method, url, headers, body = "GET", "", {}, None
    # Strip comments and collapse line continuations
    lines = curl_template.strip().split("\n")
    lines = [l for l in lines if not l.strip().startswith("#")]
    curl = " ".join(lines).replace("\\\n", " ").replace("\\", "").strip()
    if curl.startswith("curl "):
        curl = curl[5:]
    m_match = re.search(r'-X\s+(\w+)', curl)
    if m_match:
        method = m_match.group(1).upper()
    h_matches = re.findall(r"""-H\s+['"](.+?)['"]""", curl)
    for h in h_matches:
        k, *v = h.split(":")
        headers[k.strip()] = ":".join(v).strip()
    # Match -d with balanced single or double quotes (greedy within outer quotes)
    d_match = re.search(r"""(?:-d|--data|--data-raw)\s+'([^']+)'""", curl)
    if not d_match:
        d_match = re.search(r'''(?:-d|--data|--data-raw)\s+"([^"]+)"''', curl)
    if d_match:
        try:
            body = json.loads(d_match.group(1))
            for k in body:
                if k in args:
                    body[k] = args[k]
        except Exception:
            body = args
        if not m_match:
            method = "POST"
    url_match = re.search(r"""(https?://[^\s'"]+)""", curl)
    if url_match:
        url = url_match.group(1).strip("'\"")
    return {"method": method, "url": url, "headers": headers, "body": body or args}


def _parse_sse_response(raw: str) -> str:
    """Parse SSE stream and extract the meaningful answer.
    
    Handles streams with answer_chunk events (assembles full answer),
    result events (extracts data), and falls back to collecting all
    non-debug event data.
    """
    answer_chunks = []
    results = []
    final_answer = ""
    
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        
        evt_type = evt.get("type", "")
        
        # Collect answer chunks (streamed final answer)
        if evt_type == "answer_chunk":
            chunk = evt.get("chunk", "")
            answer_chunks.append(chunk)
        # Capture the full answer text if provided in one shot
        elif evt_type == "answer_start":
            pass  # answer_start just signals the beginning
        elif evt_type == "answer_done":
            pass  # we already collected chunks
        # Capture query results
        elif evt_type == "result" and evt.get("success"):
            records = evt.get("records", [])
            if records:
                results.append({"query": evt.get("coql", ""), "records": records[:20]})
        # Capture followup resolution
        elif evt_type == "followup_resolved":
            resolved = evt.get("resolved", "")
            if resolved:
                final_answer = f"Resolved query: {resolved}\n"
    
    # Prefer assembled answer chunks
    if answer_chunks:
        assembled = "".join(answer_chunks).strip()
        if assembled:
            return assembled[:4000]
    
    # Fall back to result records
    if results:
        return json.dumps(results, indent=2)[:4000]
    
    # Last resort: return first meaningful data events
    return final_answer[:4000] if final_answer else raw[:4000]


def split_first_sentence(buf: str, is_first: bool = False):
    min_len = 8 if is_first else 15
    # First sentence end at or past min_len, so a short opener ("Sure.") joins
    # the next sentence instead of holding the whole reply back.
    for m in re.finditer(r'[.!?\u0964](?:\s|$)', buf):
        if m.end() >= min_len:
            sentence = buf[: m.end()].strip()
            remainder = buf[m.end() :].lstrip()
            return sentence, remainder
    if is_first and len(buf) > 40:
        m2 = re.search(r'([^,;]*[,;])\s', buf)
        if m2 and m2.end() >= 20:
            sentence = buf[: m2.end()].strip()
            remainder = buf[m2.end() :].lstrip()
            return sentence, remainder
    return None, buf


# ================================================================
# WebSocket: Voice Call
# ================================================================
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


@app.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    session = VoiceSession()
    session._ws = ws
    ACTIVE_VOICE_SESSIONS[session.session_id] = session

    partial_task = asyncio.create_task(_partial_stt_worker(ws, session))
    response_task = asyncio.create_task(_response_worker(ws, session))

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]
                if session.state in ("listening", "speaking", "processing"):
                    await _process_audio_chunk(chunk, session, ws)
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                action = msg.get("action")

                if action == "start_call":
                    session.state = "listening"
                    session.chat_history = []
                    session.audio_buffer.clear()
                    session.active_handoff_id = None
                    session.handoff_connected = False
                    session.model = ModelSettings.from_dict(msg.get("model"))
                    print(f"[ORCH] Call started with model={session.model.describe()}")
                    agent_id = msg.get("agent_id")
                    if agent_id:
                        session.apply_agent(agent_id)
                    await ws.send_json({"type": "status", "state": "listening"})

                elif action == "set_model":
                    session.model = ModelSettings.from_dict(msg.get("model"))
                    print(f"[ORCH] Model switched to {session.model.describe()}")
                    await ws.send_json({"type": "model_set", "provider": session.model.provider})

                elif action == "end_call":
                    session.state = "idle"
                    session.interrupt_event.set()
                    session.audio_buffer.clear()
                    session.active_handoff_id = None
                    session.handoff_connected = False
                    await ws.send_json({"type": "status", "state": "idle"})

                elif action == "switch_agent":
                    agent_id = msg.get("agent_id")
                    if agent_id:
                        session.apply_agent(agent_id)
                        await ws.send_json({"type": "agent_switched", "agent_id": agent_id})

                elif action == "handoff_audio":
                    handoff_id = msg.get("handoff_id") or session.active_handoff_id
                    handoff = handoff_manager.get(handoff_id) if handoff_id else None
                    if handoff and handoff.get("status") == "accepted" and handoff_id == session.active_handoff_id:
                        await _send_audio_to_operators(handoff_id, msg.get("audio", ""), int(msg.get("sample_rate") or INPUT_SAMPLE_RATE))

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect message" not in str(e):
            raise
    finally:
        session.active = False
        ACTIVE_VOICE_SESSIONS.pop(session.session_id, None)
        partial_task.cancel()
        response_task.cancel()


@app.websocket("/ws/handoffs/{handoff_id}")
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


async def _process_audio_chunk(chunk: bytes, session: VoiceSession, ws: WebSocket):
    energy = calc_energy(chunk)

    if session.state == "speaking":
        if energy > INTERRUPT_ENERGY_THRESHOLD:
            session.interrupt_frame_count += 1
        else:
            session.interrupt_frame_count = 0

        if session.interrupt_frame_count >= MIN_INTERRUPT_FRAMES:
            print(f"[ORCH] INTERRUPT - strong user speech ({session.interrupt_frame_count} frames, energy={energy:.0f})")
            session.interrupt_event.set()
            for t in session.pending_tts_tasks:
                t.cancel()
            session.pending_tts_tasks.clear()
            session.audio_buffer.clear()
            session.pre_buffer.clear()
            session.is_user_speaking = False
            session.speech_frame_count = 0
            session.interrupt_frame_count = 0
            try:
                await ws.send_json({"type": "interrupt"})
            except Exception:
                pass
            session.state = "listening"
            await ws.send_json({"type": "status", "state": "listening"})
        return

    if energy > ENERGY_THRESHOLD:
        session.interrupt_frame_count = 0
        session.silence_frames = 0
        session.speech_frame_count += 1
        if not session.is_user_speaking:
            session.is_user_speaking = True
            if session.pre_buffer:
                session.audio_buffer.extend(session.pre_buffer)
                session.pre_buffer.clear()
        session.audio_buffer.extend(chunk)
    else:
        session.interrupt_frame_count = 0
        if session.is_user_speaking:
            session.audio_buffer.extend(chunk)
            session.silence_frames += 1
            frames_for_turn_end = max(1, SILENCE_TURN_END_MS // CHUNK_MS)
            if session.silence_frames >= frames_for_turn_end:
                if session.speech_frame_count >= MIN_SPEECH_FRAMES:
                    session.turn_complete_event.set()
                else:
                    print(f"[ORCH] Discarded noise ({session.speech_frame_count} frames)")
                    session.audio_buffer.clear()
                session.is_user_speaking = False
                session.speech_frame_count = 0
        else:
            max_pre = INPUT_SAMPLE_RATE * 2 // 5
            session.pre_buffer.extend(chunk)
            if len(session.pre_buffer) > max_pre:
                session.pre_buffer = session.pre_buffer[-max_pre:]


async def _partial_stt_worker(ws: WebSocket, session: VoiceSession):
    while session.active:
        await asyncio.sleep(1.5)
        if session.model.gemini("stt") and not session.model.live_captions:
            continue  # each partial is a paid API call; opt-in via Model settings
        if (session.state == "listening" and session.is_user_speaking
                and len(session.audio_buffer) > INPUT_SAMPLE_RATE * 2):
            try:
                text = await transcribe_audio(bytes(session.audio_buffer), session.language, model=session.model)
                if text:
                    session.partial_text = text
                    await ws.send_json({"type": "stt_partial", "text": text})
            except Exception:
                pass


def _build_system_msg(session: VoiceSession, rag_context: str) -> str:
    system_msg = SYSTEM_PROMPT
    agent_instructions = session.get_agent_instructions()
    if agent_instructions:
        system_msg += "\n\n[Agent Instructions - follow these strictly]\n" + agent_instructions
    if session.language != "en":
        lang_name = LANG_NAMES.get(session.language, session.language)
        system_msg += f"\nIMPORTANT: You MUST respond in {lang_name}. All your replies should be in {lang_name}."
    if rag_context:
        system_msg += "\n\n" + rag_context
    return system_msg


async def _run_tool_calls(ws: WebSocket, session: VoiceSession, tool_calls: list) -> tuple:
    """Execute model tool calls and append their results to history.

    Returns (switched_agent_id, handoff_requested).
    """
    switched_agent_id = None
    handoff_requested = False

    # Tell frontend to pause mic and play wait audio
    await ws.send_json({"type": "tool_executing", "audio": TOOL_WAIT_AUDIO_B64 if TOOL_WAIT_AUDIO_B64 else None})

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        raw_args = tc["function"].get("arguments") or "{}"
        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        print(f"[ORCH] Tool call: {fn_name}({fn_args})")
        await ws.send_json({"type": "tool_call", "name": fn_name, "args": fn_args})

        result = await execute_tool_call(fn_name, fn_args, session=session)
        print(f"[ORCH] Tool result: {result[:200]}")
        await ws.send_json({"type": "tool_result", "name": fn_name, "result": result[:2000]})

        try:
            if json.loads(result).get("status") == "handoff_requested":
                handoff_requested = True
        except Exception:
            pass

        # Check if it was a switch_agent call
        if fn_name == "switch_agent":
            try:
                rj = json.loads(result)
                if rj.get("status") == "switched":
                    switched_agent_id = rj["agent_id"]
            except Exception:
                pass

        session.chat_history.append({
            "role": "tool", "tool_call_id": tc["id"], "content": result,
        })

    # Notify frontend about agent switch
    if switched_agent_id:
        await ws.send_json({"type": "agent_switched", "agent_id": switched_agent_id})
    return switched_agent_id, handoff_requested


def _merge_tool_call_deltas(acc: dict, deltas: list):
    """Accumulate streamed OpenAI-style tool_call deltas (keyed by index)."""
    for d in deltas or []:
        idx = d.get("index")
        if idx is None:
            idx = len(acc)
        cur = acc.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if d.get("id"):
            cur["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name") and not cur["function"]["name"]:
            cur["function"]["name"] = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, dict):
            cur["function"]["arguments"] = json.dumps(args)
        elif args:
            cur["function"]["arguments"] += args
        if d.get("extra_content"):
            cur["extra_content"] = d["extra_content"]  # Gemini thought signature; must be sent back


def _finalize_tool_calls(acc: dict) -> list:
    calls = []
    for idx in sorted(acc):
        tc = acc[idx]
        if not tc["function"]["name"]:
            continue
        tc["id"] = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
        tc["function"]["arguments"] = tc["function"]["arguments"] or "{}"
        calls.append(tc)
    return calls


async def _response_worker(ws: WebSocket, session: VoiceSession):
    while session.active:
        await session.turn_complete_event.wait()
        session.turn_complete_event.clear()
        if session.state not in ("listening", "speaking"):
            continue
        audio_data = bytes(session.audio_buffer)
        session.audio_buffer.clear()
        session.partial_text = ""
        if len(audio_data) < INPUT_SAMPLE_RATE:
            continue

        # 1. Transcribe
        session.state = "processing"
        await ws.send_json({"type": "status", "state": "processing"})
        t_pipeline = time.time()
        try:
            t0 = time.time()
            user_text = await transcribe_audio(audio_data, session.language, model=session.model)
            print(f"[ORCH] STT: {time.time()-t0:.2f}s -> '{user_text[:60]}'")
        except Exception as e:
            user_text = ""
            print(f"[ORCH] STT error: {e}")
            if isinstance(e, ProviderError):
                await ws.send_json({"type": "error", "message": f"Speech-to-text: {e}"})
        if not user_text:
            session.state = "listening"
            await ws.send_json({"type": "status", "state": "listening"})
            continue
        await ws.send_json({"type": "stt_final", "text": user_text})
        session.chat_history.append({"role": "user", "content": user_text})

        # 2. RAG
        t0 = time.time()
        rag_context = ""
        try:
            doc_scopes = session.get_agent_document_scopes()
            results = rag.retrieve(user_text, top_k=3, scopes=doc_scopes if doc_scopes else None)
            if results:
                rag_context = "\n\n[Relevant context from documents]\n" + "\n---\n".join(r["text"] for r in results)
        except Exception as e:
            print(f"[ORCH] RAG error: {e}")
        print(f"[ORCH] RAG: {time.time()-t0:.3f}s")

        # 3. Build messages with agent context
        system_msg = _build_system_msg(session, rag_context)

        # Get tools filtered to this agent's tool_ids
        agent_tool_ids = session.get_agent_tool_ids()
        tools = tool_library.to_openai_tools(agent_tool_ids if agent_tool_ids else None) or []
        if not _tool_list_has_handoff(tools):
            tools.append(get_handoff_openai_tool())
        # Auto-inject switch_agent tool if multiple agents
        if len(agent_manager.agents) > 1:
            tools.append(agent_manager.get_switch_tool())
        tools = tools or None

        if tools:
            system_msg += ("\n\nYou have access to tools/APIs. Use them when the user's request matches a tool's purpose. "
                          "After receiving tool results, give a natural conversational summary - do NOT repeat raw data.")

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(session.chat_history[-20:])

        # 4. LLM - check for tool calls. Gemini skips this round trip: the first
        # streamed call carries the tools, so speech starts with the first sentence.
        tool_call_response = None
        stream_tools = None  # Gemini needs the tool declarations whenever history holds tool calls
        gemini_stream_tools = tools if session.model.gemini("llm") else None
        if tools and not gemini_stream_tools:
            try:
                t0 = time.time()
                llm_resp = await call_llm_with_tools(messages, tools, model=session.model)
                choice = llm_resp["choices"][0]
                msg_obj = choice["message"]
                finish_reason = choice.get("finish_reason", "")
                print(f"[ORCH] LLM (tool check): {time.time()-t0:.2f}s, finish={finish_reason}")

                if msg_obj.get("tool_calls"):
                    session.chat_history.append(msg_obj)
                    switched_agent_id, handoff_requested = await _run_tool_calls(ws, session, msg_obj["tool_calls"])

                    if handoff_requested:
                        session.state = "handoff_waiting"
                        continue

                    # Rebuild messages for NL reply
                    if switched_agent_id:
                        system_msg = _build_system_msg(session, rag_context)

                    messages = [{"role": "system", "content": system_msg}]
                    messages.extend(session.chat_history[-20:])
                    if session.model.gemini("llm"):
                        stream_tools = tools
                    tools = None

                    # Tell frontend tool execution done, resume mic
                    await ws.send_json({"type": "tool_done"})

                elif msg_obj.get("content"):
                    raw_content = msg_obj["content"]
                    raw_fn, raw_args = _parse_raw_tool_call(raw_content)
                    if raw_fn and raw_args:
                        print(f"[ORCH] Fallback raw tool call: {raw_fn}({raw_args})")
                        name_map = tool_library.get_name_mapping()
                        display_name = name_map.get(raw_fn, raw_fn)
                        handoff_requested = False
                        await ws.send_json({"type": "tool_executing", "audio": TOOL_WAIT_AUDIO_B64 if TOOL_WAIT_AUDIO_B64 else None})
                        await ws.send_json({"type": "tool_call", "name": display_name, "args": raw_args})
                        result = await execute_tool_call(raw_fn, raw_args, session=session)
                        print(f"[ORCH] Tool result: {result[:200]}")
                        await ws.send_json({"type": "tool_result", "name": display_name, "result": result[:2000]})
                        try:
                            if json.loads(result).get("status") == "handoff_requested":
                                handoff_requested = True
                        except Exception:
                            pass

                        if raw_fn == "switch_agent":
                            try:
                                rj = json.loads(result)
                                if rj.get("status") == "switched":
                                    await ws.send_json({"type": "agent_switched", "agent_id": rj["agent_id"]})
                            except Exception:
                                pass

                        session.chat_history.append({"role": "assistant", "content": raw_content})
                        session.chat_history.append({"role": "user", "content": f"[Tool '{display_name}' returned]: {result}\n\nNow give a brief natural language summary of the result to the user. Do NOT output any tool calls."})
                        messages = [{"role": "system", "content": system_msg}]
                        messages.extend(session.chat_history[-20:])
                        tools = None
                        if handoff_requested:
                            session.state = "handoff_waiting"
                            continue
                        await ws.send_json({"type": "tool_done"})
                    else:
                        tool_call_response = raw_content
            except Exception as e:
                print(f"[ORCH] Tool-check LLM error: {e}")
                if isinstance(e, ProviderError):
                    await ws.send_json({"type": "error", "message": f"LLM: {e}"})
                    session.state = "listening"
                    await ws.send_json({"type": "status", "state": "listening"})
                    continue

        # 5. Stream LLM -> sentence -> TTS
        session.state = "speaking"
        session.interrupt_event.clear()
        session.pending_tts_tasks = []
        await ws.send_json({"type": "status", "state": "speaking"})

        full_response = ""
        sentence_buf = ""
        is_first_sentence = True
        t_llm_start = time.time()
        first_token_time = None
        tts_lang_code = KOKORO_LANG_CODES.get(session.language, "a")

        tts_condition = asyncio.Condition()
        # seq -> {"chunks": [pcm, ...], "done": bool, "text": str, "t0": float, "first": float|None}
        tts_entries: dict[int, dict] = {}
        tts_state = {"issued": 0, "closed": False, "error_sent": False, "first_audio": False}

        async def _tts_and_store(sequence: int, text: str, voice: str, speed: float):
            entry = tts_entries[sequence]
            try:
                if session.interrupt_event.is_set():
                    return
                async for chunk in synthesize_speech_stream(
                    text, voice, speed,
                    lang_code=tts_lang_code,
                    language=session.language,
                    model=session.model,
                ):
                    if session.interrupt_event.is_set():
                        return
                    async with tts_condition:
                        if entry["first"] is None:
                            entry["first"] = time.time()
                        entry["chunks"].append(chunk)
                        tts_condition.notify_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ORCH] TTS failed for seq={sequence}: {exc}")
                if isinstance(exc, ProviderError) and not tts_state["error_sent"]:
                    tts_state["error_sent"] = True
                    try:
                        await ws.send_json({"type": "error", "message": f"Text-to-speech: {exc}"})
                    except Exception:
                        pass
            finally:
                async with tts_condition:
                    entry["done"] = True
                    tts_condition.notify_all()

        def _queue_tts(text: str):
            text = (text or "").strip()
            if not text:
                return
            sequence = tts_state["issued"]
            tts_state["issued"] += 1
            tts_entries[sequence] = {"chunks": [], "done": False, "text": text, "t0": time.time(), "first": None}
            task = asyncio.create_task(_tts_and_store(sequence, text, session.voice, session.speed))
            session.pending_tts_tasks.append(task)

        async def _tts_ordered_sender():
            # Sentences synthesize in parallel; audio is forwarded strictly in
            # order, chunk by chunk, as soon as the current sentence produces it.
            next_sequence = 0
            while True:
                async with tts_condition:
                    await tts_condition.wait_for(
                        lambda: session.interrupt_event.is_set()
                        or (next_sequence in tts_entries
                            and (tts_entries[next_sequence]["chunks"] or tts_entries[next_sequence]["done"]))
                        or (tts_state["closed"] and next_sequence >= tts_state["issued"])
                    )
                    if session.interrupt_event.is_set():
                        return
                    if next_sequence not in tts_entries:
                        return  # closed and everything sent
                    entry = tts_entries[next_sequence]
                    chunks, entry["chunks"] = entry["chunks"], []
                    finished = entry["done"] and not entry["chunks"]
                    current_sequence = next_sequence
                for chunk in chunks:
                    if session.interrupt_event.is_set():
                        return
                    await ws.send_json({"type": "tts_audio", "audio": base64.b64encode(chunk).decode(), "seq": current_sequence})
                    if not tts_state["first_audio"]:
                        tts_state["first_audio"] = True
                        print(f"[ORCH] First audio: {time.time() - t_pipeline:.2f}s after end of speech "
                              f"(+{SILENCE_TURN_END_MS / 1000:.1f}s silence detection)")
                if finished:
                    if entry["first"] is not None:
                        print(f"[ORCH] TTS seq={current_sequence} '{entry['text'][:40]}...' -> "
                              f"first chunk {entry['first'] - entry['t0']:.2f}s, done {time.time() - entry['t0']:.2f}s")
                    tts_entries.pop(current_sequence, None)
                    next_sequence += 1

        session.pending_tts_tasks.append(asyncio.create_task(_tts_ordered_sender()))

        async def _stream_pass(pass_messages: list, pass_tools: Optional[list], tool_choice: Optional[str]) -> tuple:
            """Stream one LLM completion into the UI + TTS. Returns (interrupted, tool_calls)."""
            nonlocal full_response, sentence_buf, is_first_sentence, first_token_time
            tool_acc: dict = {}
            thinking_buffer = ""
            in_thinking = True
            async for token_or_tc in stream_llm_tokens(pass_messages, pass_tools, model=session.model, tool_choice=tool_choice):
                if isinstance(token_or_tc, dict) and "tool_calls" in token_or_tc:
                    _merge_tool_call_deltas(tool_acc, token_or_tc["tool_calls"])
                    continue
                token = token_or_tc
                if in_thinking:
                    thinking_buffer += token
                    stripped = thinking_buffer.lstrip()
                    if stripped.startswith("<start_of_thought>"):
                        if "<end_of_thought>" in stripped:
                            after = stripped.split("<end_of_thought>", 1)[1].lstrip()
                            thinking_buffer = ""
                            in_thinking = False
                            if after:
                                token = after
                            else:
                                continue
                        else:
                            continue
                    elif stripped.lower().startswith("thought"):
                        rest = stripped[7:].lstrip()
                        thinking_buffer = ""
                        in_thinking = False
                        if rest:
                            token = rest
                        else:
                            continue
                    elif len(stripped) > 10:
                        token = thinking_buffer
                        thinking_buffer = ""
                        in_thinking = False
                    else:
                        continue

                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[ORCH] LLM TTFT: {first_token_time - t_llm_start:.3f}s")

                if session.interrupt_event.is_set():
                    print("[ORCH] LLM stream interrupted by user")
                    return True, []

                full_response += token
                sentence_buf += token
                await ws.send_json({"type": "llm_token", "token": token})
                sentence, sentence_buf = split_first_sentence(sentence_buf, is_first_sentence)
                if sentence:
                    is_first_sentence = False
                    _queue_tts(sentence)

            if in_thinking and thinking_buffer.strip() and not thinking_buffer.lstrip().startswith("<start_of_thought>"):
                # Short reply that never passed the thought-prefix check.
                full_response += thinking_buffer
                sentence_buf += thinking_buffer
                await ws.send_json({"type": "llm_token", "token": thinking_buffer})
            if sentence_buf.strip() and not session.interrupt_event.is_set():
                await ws.send_json({"type": "llm_token", "token": ""})
                _queue_tts(sentence_buf.strip())
                sentence_buf = ""
            return False, _finalize_tool_calls(tool_acc)

        interrupted = False
        history_saved = False
        try:
            if tool_call_response:
                full_response = tool_call_response
                stripped = full_response.lstrip()
                if stripped.lower().startswith("thought"):
                    full_response = stripped[7:].lstrip()
                if "<start_of_thought>" in full_response and "<end_of_thought>" in full_response:
                    full_response = full_response.split("<end_of_thought>", 1)[1].lstrip()
                await ws.send_json({"type": "llm_token", "token": full_response})
                remaining = full_response
                while remaining:
                    sent, remaining = split_first_sentence(remaining, is_first_sentence)
                    if sent:
                        is_first_sentence = False
                        _queue_tts(sent)
                    else:
                        if remaining.strip():
                            _queue_tts(remaining.strip())
                        break
            elif gemini_stream_tools:
                # Gemini: one streamed call with tools. Text is spoken as it
                # arrives; if the model calls tools, run them and stream the reply.
                t0 = time.time()
                interrupted, tool_calls = await _stream_pass(messages, gemini_stream_tools, None)
                print(f"[ORCH] LLM stream: {time.time()-t0:.2f}s, tool_calls={len(tool_calls)}")
                if tool_calls and not interrupted:
                    session.chat_history.append({"role": "assistant", "content": full_response or None, "tool_calls": tool_calls})
                    history_saved = True
                    switched_agent_id, handoff_requested = await _run_tool_calls(ws, session, tool_calls)
                    if handoff_requested:
                        session.state = "handoff_waiting"
                    else:
                        if switched_agent_id:
                            system_msg = _build_system_msg(session, rag_context)
                        messages = [{"role": "system", "content": system_msg}]
                        messages.extend(session.chat_history[-20:])
                        await ws.send_json({"type": "tool_done"})
                        reply_start = len(full_response)
                        interrupted, _ = await _stream_pass(messages, gemini_stream_tools, "none")
                        if full_response[reply_start:].strip():
                            session.chat_history.append({"role": "assistant", "content": full_response[reply_start:]})
            else:
                interrupted, _ = await _stream_pass(messages, stream_tools, "none")

            async with tts_condition:
                tts_state["closed"] = True
                tts_condition.notify_all()

            if session.pending_tts_tasks and interrupted:
                for task in session.pending_tts_tasks:
                    task.cancel()
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)
            elif session.pending_tts_tasks:
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)

            int_str = "(interrupted)" if interrupted else ""
            print(f"[ORCH] Pipeline: {time.time()-t_llm_start:.2f}s {int_str}")
        except Exception as e:
            async with tts_condition:
                tts_state["closed"] = True
                tts_condition.notify_all()
            for task in session.pending_tts_tasks:
                if not task.done():
                    task.cancel()
            if session.pending_tts_tasks:
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)
            print(f"[ORCH] LLM/TTS pipeline error: {e}")
            await ws.send_json({"type": "error", "message": str(e)})
        finally:
            session.pending_tts_tasks = []

        if full_response and not history_saved:
            session.chat_history.append({"role": "assistant", "content": full_response})

        await ws.send_json({"type": "llm_done"})
        if session.state == "speaking":
            session.state = "listening"
            session.interrupt_event.clear()
            await ws.send_json({"type": "status", "state": "listening"})


# ================================================================
# REST: Documents
# ================================================================
def _request_model(request: Request) -> ModelSettings:
    """Model settings the browser attached for background LLM work (K-Quest generation)."""
    return ModelSettings.from_header(request.headers.get("x-model-settings"))


@app.post("/api/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...), relative_path: Optional[str] = Form(None)):
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")
    try:
        doc_id = rag.add_document(file.filename or "document.txt", content, relative_path=relative_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    asyncio.create_task(rag.generate_faqs(doc_id, model=_request_model(request)))
    doc = rag.documents.get(doc_id, {})
    return JSONResponse({"status": "ok", "document": doc})


@app.post("/api/documents/ingest")
async def ingest_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    paths: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    parsed_paths = []
    if paths:
        try:
            parsed_paths = json.loads(paths)
        except Exception:
            raise HTTPException(400, "paths must be a JSON array")
        if not isinstance(parsed_paths, list):
            raise HTTPException(400, "paths must be a JSON array")

    file_items = []
    for index, file in enumerate(files):
        content = await file.read()
        if not content:
            continue
        rel_path = parsed_paths[index] if index < len(parsed_paths) else (file.filename or f"document-{index + 1}")
        for item in rag.expand_archive(file.filename or "archive.zip", content, rel_path):
            file_items.append(item)
    if not file_items:
        raise HTTPException(400, "No non-empty files uploaded")

    collection_id = rag.create_collection(collection_name or "Upload", len(file_items))
    job_id = rag.create_job(collection_id, len(file_items))
    asyncio.create_task(rag.process_ingest_job(job_id, file_items, model=_request_model(request)))
    return JSONResponse({
        "status": "queued",
        "job_id": job_id,
        "collection_id": collection_id,
        "file_count": len(file_items),
    })


@app.get("/api/documents")
async def list_documents():
    return JSONResponse({"documents": rag.list_documents()})


@app.get("/api/documents/tree")
async def documents_tree():
    return JSONResponse({"tree": rag.list_tree(), "collections": rag.list_collections()})


@app.get("/api/documents/jobs")
async def list_document_jobs():
    return JSONResponse({"jobs": rag.list_jobs()})


@app.get("/api/documents/jobs/{job_id}")
async def document_job(job_id: str):
    job = rag.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse({"job": job})


@app.post("/api/documents/collections/{collection_id}/retry")
async def retry_collection_quests(collection_id: str, request: Request):
    """Re-queue K-Quest generation for all ready files whose quest_status is still pending."""
    job_id = rag.retry_collection(collection_id)
    if not job_id:
        return JSONResponse({"status": "nothing_to_retry", "message": "All files already have K-Quest data"})
    asyncio.create_task(rag.run_retry_quests(job_id, collection_id, model=_request_model(request)))
    return JSONResponse({"status": "queued", "job_id": job_id})


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = rag.get_document_view(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return JSONResponse(doc)


@app.delete("/api/documents/collections/{collection_id}")
async def delete_document_collection(collection_id: str):
    ok = rag.delete_collection(collection_id)
    if not ok:
        raise HTTPException(404, "Upload not found")
    return JSONResponse({"status": "deleted"})


@app.delete("/api/documents/folders")
async def delete_document_folder(path: str):
    ok = rag.delete_folder(path)
    if not ok:
        raise HTTPException(404, "Folder not found")
    return JSONResponse({"status": "deleted"})


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    ok = rag.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "Document not found")
    return JSONResponse({"status": "deleted"})


@app.get("/api/rag/search")
async def rag_search(q: str, top_k: int = 5, scopes: Optional[str] = None):
    doc_scopes = None
    if scopes:
        try:
            doc_scopes = json.loads(scopes)
        except Exception:
            raise HTTPException(400, "scopes must be JSON")
    results = rag.retrieve(q, top_k, scopes=doc_scopes)
    return JSONResponse({"results": results})


# ================================================================
# REST: Tool Library
# ================================================================
@app.get("/api/tools")
async def list_tools():
    return JSONResponse({"tools": tool_library.list_all()})


@app.post("/api/tools")
async def add_tool(data: dict):
    if not data.get("name"):
        raise HTTPException(400, "Name is required")
    if not data.get("description"):
        raise HTTPException(400, "Description is required")
    tool = tool_library.add(data)
    return JSONResponse({"status": "ok", "tool": tool})


@app.get("/api/tools/{tool_id}")
async def get_tool(tool_id: str):
    tool = tool_library.get(tool_id)
    if not tool:
        raise HTTPException(404, "Tool not found")
    return JSONResponse(tool)


@app.put("/api/tools/{tool_id}")
async def update_tool(tool_id: str, data: dict):
    tool = tool_library.update(tool_id, data)
    if not tool:
        raise HTTPException(404, "Tool not found")
    return JSONResponse({"status": "ok", "tool": tool})


@app.delete("/api/tools/{tool_id}")
async def delete_tool(tool_id: str):
    if not tool_library.delete(tool_id):
        raise HTTPException(404, "Tool not found")
    return JSONResponse({"status": "deleted"})


# ================================================================
# REST: Agent Management
# ================================================================
@app.get("/api/agents")
async def list_agents():
    return JSONResponse({"agents": agent_manager.list_all()})


@app.post("/api/agents")
async def create_agent(data: dict):
    if not data.get("name"):
        raise HTTPException(400, "Name is required")
    agent = agent_manager.add(data)
    return JSONResponse({"status": "ok", "agent": agent})


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = agent_manager.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return JSONResponse(agent)


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, data: dict):
    agent = agent_manager.update(agent_id, data)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return JSONResponse({"status": "ok", "agent": agent})


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if not agent_manager.delete(agent_id):
        raise HTTPException(404, "Agent not found")
    return JSONResponse({"status": "deleted"})


# ================================================================
# REST: Human Handoffs
# ================================================================
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


@app.get("/api/handoffs")
async def list_handoffs(status: Optional[str] = None):
    return JSONResponse({"handoffs": handoff_manager.list_all(status)})


@app.get("/api/handoffs/{handoff_id}")
async def get_handoff(handoff_id: str):
    handoff = handoff_manager.get(handoff_id)
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    return JSONResponse({"handoff": handoff})


@app.post("/api/handoffs/{handoff_id}/accept")
async def accept_handoff(handoff_id: str, data: Optional[dict] = None):
    operator_name = (data or {}).get("operator_name") or "Human Agent"
    handoff = handoff_manager.update_status(handoff_id, "accepted", operator_name=operator_name)
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})


@app.post("/api/handoffs/{handoff_id}/resolve")
async def resolve_handoff(handoff_id: str, data: Optional[dict] = None):
    handoff = handoff_manager.update_status(handoff_id, "resolved", note=(data or {}).get("note", ""))
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})


@app.post("/api/handoffs/{handoff_id}/cancel")
async def cancel_handoff(handoff_id: str, data: Optional[dict] = None):
    handoff = handoff_manager.update_status(handoff_id, "cancelled", note=(data or {}).get("note", ""))
    if not handoff:
        raise HTTPException(404, "Handoff not found")
    await _notify_handoff_session(handoff)
    return JSONResponse({"handoff": handoff})


# ================================================================
# REST: Model provider (Gemini BYOK)
# ================================================================
@app.get("/api/model/defaults")
async def model_defaults():
    return JSONResponse({
        "llm_model": GEMINI_LLM_MODEL,
        "stt_model": GEMINI_STT_MODEL,
        "tts_model": GEMINI_TTS_MODEL,
        "reasoning_effort": GEMINI_REASONING_EFFORT,
        "voices": GEMINI_VOICES,
        "server_key": bool(GEMINI_API_KEY),
        "mode": MODEL_PROVIDER,
    })


@app.post("/api/model/test")
async def model_test(data: dict):
    """Check a Gemini key from this server (also proves outbound access) and list models."""
    try:
        models = await gemini_list_models(ModelSettings.from_dict({**(data or {}), "provider": "gemini"}))
    except ProviderError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except httpx.HTTPError as e:
        return JSONResponse({"ok": False, "error": f"Could not reach the Gemini API from this server: {e}"}, status_code=502)
    return JSONResponse({"ok": True, **models})


# ================================================================
# Health / Status
# ================================================================
@app.get("/health")
async def health():
    if GEMINI_ONLY:
        return {"status": "ok", "mode": MODEL_PROVIDER, "services": {}}
    return {"status": "ok", "mode": MODEL_PROVIDER, "services": {"stt": STT_URL, "llm": LLM_URL, "tts": TTS_URL}}


@app.get("/api/services/status")
async def services_status():
    status = {}
    if GEMINI_ONLY:
        return JSONResponse({"mode": MODEL_PROVIDER, "services": status})
    for name, url in [("stt", STT_URL), ("llm", LLM_URL), ("tts", TTS_URL), ("indic", INDIC_URL)]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{url}/health")
                status[name] = "online" if r.status_code == 200 else "error"
        except Exception:
            status[name] = "offline"
    return JSONResponse({"mode": MODEL_PROVIDER, "services": status})


# ================================================================
# Static Files
# ================================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/human")
async def serve_human_console():
    return FileResponse(str(STATIC_DIR / "human.html"))


if __name__ == "__main__":
    import os
    cert = Path(__file__).parent / "cert.pem"
    key = Path(__file__).parent / "key.pem"
    ssl_kw = {}
    # HTTPS=auto (default) uses cert.pem/key.pem when present; HTTPS=0 forces plain HTTP.
    if os.getenv("HTTPS", "auto").lower() not in ("0", "false", "no", "off") and cert.exists() and key.exists():
        ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        print("[ORCH] HTTPS enabled (self-signed cert)")
    print(f"[ORCH] Model provider mode: {MODEL_PROVIDER}" + (" (Gemini only, no local GPU services)" if GEMINI_ONLY else ""))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")), **ssl_kw)
