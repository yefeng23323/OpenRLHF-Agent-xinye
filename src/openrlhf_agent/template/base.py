from typing import Any, Dict, List, Optional, Tuple, Union
from openrlhf_agent.utils.types import ToolCall

class Template:

    def render_system(self, text: str, tools: List[str]) -> str:
        raise NotImplementedError
    
    def render_turn(self, role: str, text: str) -> str:
        raise NotImplementedError
    
    def render_generation_prompt(self, role: str):
        raise NotImplementedError

    def render_messages(
        self,
        system_prompt: str,
        history: List[Dict[str,str]],
        tools_manifest: Optional[List[Dict[str,Any]]]=None
    ) -> Dict[str,Any]:
        raise NotImplementedError
    
    def extract_tool_calls_from_text(self, text: str) -> Tuple[bool, Union[List[ToolCall], None]]:
        raise NotImplementedError
