"""Agent definitions (data/agents.json) and the auto-injected switch_agent tool."""

import json
import uuid
from typing import Optional


from app.config import DATA_DIR




AGENTS_FILE = DATA_DIR / "agents.json"


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


agent_manager = AgentManager()
