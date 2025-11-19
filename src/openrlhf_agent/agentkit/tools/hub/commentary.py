"""Friendly commentary tool the model uses to keep the user updated."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..base import ToolBase


class CommentaryTool(ToolBase):
    """Lightweight tool for quick, action-focused updates."""

    name = "commentary"
    description = (
        "Cheerfully tell the user what you are actively doing right now—keep it short, specific, and action-focused."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "One sentence that starts with a verb and states the concrete task you are tackling right now.",
            },
        },
        "required": ["status"],
    }

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        status = self._coerce_text(arguments.get("status"))
        if not status:
            raise ValueError("`status` must be a non-empty string.")

        payload: Dict[str, Any] = {
            "ok": True,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


__all__ = ["CommentaryTool"]
