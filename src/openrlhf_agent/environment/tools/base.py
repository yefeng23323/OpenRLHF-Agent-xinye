from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

class ToolBase(BaseModel):
    name: str
    description: str
    parameters: List[Dict[str, Any]]

    model_config = {"arbitrary_types_allowed": True}

    def openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def call(self, context: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError("Tool must implement call()")
