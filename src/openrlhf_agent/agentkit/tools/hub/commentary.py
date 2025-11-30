"""Friendly commentary tool the model uses to keep the user updated."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class CommentaryTool(ToolBase):
    """Lightweight tool for quick, action-focused updates."""

    name = "commentary"
    description = "Provide a status update or progress report to the user."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to display to the user.",
            },
        },
        "required": ["message"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return json.dumps({"ok": True}, ensure_ascii=False)
