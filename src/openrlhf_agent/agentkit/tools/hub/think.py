"""Built-in think tool."""

from __future__ import annotations

from typing import Any, Dict

from openrlhf_agent.agentkit.tools import ToolBase


class ThinkTool(ToolBase):
    name = "think"
    description = "Structured thinking tool for the model to capture reasoning, plans, and intermediate calculations."
    parameters: Dict[str, object] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "step-by-step reasoning, a concise plan, and intermediate calculations.",
            }
        },
        "required": ["note"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return ""
