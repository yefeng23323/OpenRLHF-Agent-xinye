from typing import Any, Dict, List
from ..base import ToolBase


class FinalTool(ToolBase):
    """
    Submit the final user-facing answer and terminate the episode.
    """
    name: str = "final"
    description: str = (
        "Return the final user-facing answer and end the task. "
        "Only call this after sufficient planning or a recent think(...) step. "
        "Do not include meta text or tool wrappers in the answer."
    )
    parameters: List[Dict[str, Any]] = [
        {"name": "answer", "type": "string", "description": "The final user-facing answer only.", "required": True},
    ]
    # parameters: Dict[str, Any] = {
    #     "type": "object",
    #     "properties": {"answer": {"type": "string"}},
    #     "required": ["answer"],
    # }

    def call(self, context: Dict[str, Any], **kwargs) -> str:
        answer = (kwargs.get("answer", "")).strip()
        return answer
