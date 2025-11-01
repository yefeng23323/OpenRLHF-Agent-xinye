from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any]

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False

class TraceStep(BaseModel):
    kind: Literal["model","tool"]
    payload: Dict[str, Any]


