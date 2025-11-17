"""Tool used to emit the final visible answer."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class FinalTool(ToolBase):
    """Explicit final-answer tool for structured outputs."""

    name = "final"
    description = "Return the final answer that will be shown to the user."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Plain-text answer for the user.",
            }
        },
        "required": ["answer"],
    }

    def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        answer = str(arguments.get("answer", "")).strip()
        if not answer:
            return json.dumps(
                {"ok": False, "error": "answer must be a non-empty string."},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "answer": answer}, ensure_ascii=False)


__all__ = ["FinalTool"]
