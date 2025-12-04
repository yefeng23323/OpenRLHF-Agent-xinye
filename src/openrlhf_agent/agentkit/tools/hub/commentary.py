"""Friendly commentary tool the model uses to keep the user updated."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class CommentaryTool(ToolBase):
    """Tool for short progress updates, not for solving the task."""

    # Tool name used in tool_calls
    name = "commentary"

    # High-level description used by the model to decide when to call it
    description = (
        "Send a brief, friendly status update about what you are doing. "
        "Use this tool only to describe your current actions or progress, "
        "not to give the final answer or important content."
    )

    # JSON schema for arguments
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "A short status message about your current action, e.g. "
                    "\"Thinking through the steps\", "
                    "\"Checking recent data\", "
                    "\"Reviewing your code for bugs\". "
                    "Do NOT include the final answer, key conclusions, or long explanations here."
                ),
            },
        },
        "required": ["message"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        # The backend does nothing special, it just confirms the update.
        # The user-facing value is the message itself in the assistant step.
        return json.dumps({"ok": True}, ensure_ascii=False)