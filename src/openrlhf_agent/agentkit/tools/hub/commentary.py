"""Commentary tool for short progress updates."""

from __future__ import annotations

import json
from typing import Any, Dict

from openrlhf_agent.agentkit.tools import ToolBase


class CommentaryTool(ToolBase):
    """Send a brief status update separate from the final answer."""

    name = "commentary"
    description = (
        "Send a short status update about current actions or progress. "
        "Do not use this tool for the final answer or key content."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "Short status message about the current action, e.g. "
                    "\"Checking recent data\", \"Reviewing code\". "
                    "Do not include final answers or long explanations."
                ),
            },
        },
        "required": ["message"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return json.dumps({"ok": True}, ensure_ascii=False)
