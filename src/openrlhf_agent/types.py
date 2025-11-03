from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolCall:
    """Structured representation of a tool call emitted by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Container for tool execution output that is fed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False
