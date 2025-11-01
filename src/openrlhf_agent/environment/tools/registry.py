from __future__ import annotations
from typing import Dict, Iterable, List
from .base import ToolBase

class ToolRegistry:
    def __init__(self, tools):
        self._tools: Dict[str, ToolBase] = {
            tool.name: tool for tool in tools
        }

    def register(self, tool: ToolBase) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolBase:
        return self._tools[name]

    def list_openai_tools(self) -> List[dict]:
        return [t.openai_tool() for t in self._tools.values()]

    def names(self) -> Iterable[str]:
        return list(self._tools.keys())
