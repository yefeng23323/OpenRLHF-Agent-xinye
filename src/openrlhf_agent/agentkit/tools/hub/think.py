"""Built-in hidden think tool."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class ThinkTool(ToolBase):
    """Hidden planning tool used to capture private notes."""

    name = "think"
    description = "Write down private notes before taking a visible action."
    parameters: Dict[str, object] = {
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "Short plan or reasoning that stays internal.",
            }
        },
        "required": ["notes"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return json.dumps({"ok": True}, ensure_ascii=False)
