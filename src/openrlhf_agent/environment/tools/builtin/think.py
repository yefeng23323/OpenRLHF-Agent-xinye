import json
from typing import Any, Dict, List
from ..base import ToolBase


class ThinkTool(ToolBase):
    """
    Record intermediate thoughts that will be shown to the user.
    Use this to outline goals, known facts, uncertainties, and next steps.
    """
    name: str = "think"
    description: str = (
        "Share a concise, user-visible checkpoint: goal, knowns, unknowns, and next steps. "
        "Do not reveal step-by-step chain-of-thought; keep it brief and factual. "
        "Use this when information is missing or planning is needed, typically before final."
    )
    parameters: List[Dict[str, Any]] = [
        {"name": "notes", "type": "string", "description": "A short summary of the current goal, key facts, open questions, and next steps.", "required": True},
    ]
    # parameters: Dict[str, Any] = {
    #     "type": "object",
    #     "properties": {"note": {"type": "string"}},
    #     "required": ["note"],
    # }
    
    def call(self, context: Dict[str, Any], **kwargs) -> str:
        # notes = kwargs.get("notes", "")
        observation = json.dumps(
            {
                "ok": True,
                # "notes_char_len": len(notes),
                # "message": "Note stored. Keep using think until ready. Call final to finish.",
            },
            ensure_ascii=False,
        )
        return observation
