import re
import json

from typing import List, Optional, Dict, Any, Tuple, Union

from .base import Template
from openrlhf_agent.utils.types import ToolCall

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

QWEN3_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

QWEN3_SYSTEM_WITH_TOOLS_TEMPLATE = """
{system}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".strip()


class Qwen3Template(Template):
    def render_generation_prompt(self, role: str = "assistent"):
        return f"{IM_START}{role}\n"

    def render_turn(self, role: str, text: str, add_generation_prompt=False) -> str:
        turn = f"{IM_START}{role}\n{text}{IM_END}\n"
        if add_generation_prompt:
            turn += self.render_generation_prompt()
        return turn

    def render_system(
        self,
        text: str,
        tools_manifest: Optional[List[Dict[str,Any]]] = None
    ) -> str:
        if tools_manifest:
            text = QWEN3_SYSTEM_WITH_TOOLS_TEMPLATE.format(
                system=text,
                tools='\n'.join([json.dumps(i, ensure_ascii=False) for i in tools_manifest])
            )
        return self.render_turn(role="system", text=text)

    def render_tool_response(self, text: str):
        return f"<tool_response>\n{text}\n</tool_response>"

    def render_messages(
        self,
        history: List[Dict[str,str]],
        tools_manifest: Optional[List[Dict[str,Any]]] = None,
        add_generation_prompt=False,
    ) -> Dict[str,Any]:
        s = ""
        
        if history and history[0]["role"] == "system":
            s += self.render_system(
                history[0]["content"],
                tools_manifest=tools_manifest
            )
            history.pop(0)
        
        for item in history:
            role = item["role"]
            content = item["content"]
            s += self.render_turn(role, content)
        
        if add_generation_prompt:
            s += self.render_generation_prompt()
        
        return s
    
    def extract_tool_calls_from_text(self, text: str) -> Tuple[bool, Union[List[ToolCall], None]]:
        matches = QWEN3_TOOL_RE.findall(text or "")
        if len(matches) < 1:
            return True, []

        calls = list()
        for i, m in enumerate(matches):
            try:
                obj = json.loads(m)
                name = obj.get("name")
                if not isinstance(name, str) or not name:
                    calls.append(None)
                    continue

                arguments = obj.get("arguments", {}) or {}
                if not isinstance(arguments, dict):
                    calls.append(None)
                    continue

                calls.append(ToolCall(
                    id=obj.get("id") or f"call_{i}",
                    name=name,
                    arguments=arguments,
                ))
            except Exception:
                calls.append(None)
                continue
        
        return False, calls
