"""Tool abstractions plus the default built-in tool set."""

from openrlhf_agent.environment.tools.base import ToolBase
from openrlhf_agent.environment.tools.registry import ToolRegistry
from openrlhf_agent.environment.tools.think import ThinkTool
from openrlhf_agent.environment.tools.final import FinalTool

_TOOLS_ = [
    "ThinkTool", "FinalTool"
]

__all__ = ["ToolBase", "ToolRegistry", *_TOOLS_]
