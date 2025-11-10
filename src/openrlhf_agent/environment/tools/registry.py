"""Simple runtime registry for tools."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from openrlhf_agent.environment.tools.base import ToolBase


class ToolRegistry:
    """Simple name-to-tool lookup table."""

    def __init__(self, tools: Sequence[ToolBase]):
        self._tools: Dict[str, ToolBase] = {tool.name: tool for tool in tools}

    def register(self, tool: ToolBase) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tools.values()]

    def get(self, name: str) -> ToolBase:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name]


__all__ = ["ToolRegistry"]
