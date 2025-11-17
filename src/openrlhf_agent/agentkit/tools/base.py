"""Abstract base class for environment tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ToolBase(ABC):
    """Minimal function-style tool definition."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def openai_tool(self) -> Dict[str, Any]:
        """Return a schema that matches OpenAI's function tool format."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        """Execute the tool and return a string payload."""

        raise NotImplementedError


__all__ = ["ToolBase"]
