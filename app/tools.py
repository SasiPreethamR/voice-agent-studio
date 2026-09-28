"""API tool library (data/tools.json) and tool execution: mock responses,
realtime HTTP calls parsed from cURL, switch_agent and human handoff."""

import re
import json
import uuid
from typing import Optional

import httpx

from app.config import DATA_DIR

from app.agents import agent_manager
from app.handoff import _is_handoff_tool_name, handoff_manager



TOOLS_FILE = DATA_DIR / "tools.json"


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


tool_library = ToolLibrary()


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
