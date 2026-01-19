"""Tool used to emit the final visible answer."""

from __future__ import annotations

import json
from typing import Any, Dict

from openrlhf_agent.agentkit.tools import ToolBase


class FinalTool(ToolBase):
    """Explicit final-answer tool for structured outputs."""

    name = "final"
    description = "Return the final response that will be shown to the user."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Final response to the user.",
            },
        },
        "required": ["response"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        response = str(arguments.get("response", "")).strip()
        if not response:
            return json.dumps(
                {"ok": False, "error": "response must be a non-empty string."},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "response": response}, ensure_ascii=False)
