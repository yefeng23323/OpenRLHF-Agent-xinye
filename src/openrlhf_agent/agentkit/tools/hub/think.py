"""Built-in hidden think tool."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class ThinkTool(ToolBase):
    """Hidden planning tool used to capture private note."""

    name = "think"
    description = "Write down private note before taking a visible action."
    parameters: Dict[str, object] = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Short plan or reasoning that stays internal.",
            }
        },
        "required": ["note"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        return ""
