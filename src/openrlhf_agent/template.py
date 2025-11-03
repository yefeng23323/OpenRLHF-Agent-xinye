import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .types import ToolCall

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


class Template(ABC):
    @abstractmethod
    def render_system(self, text: str, tools_manifest: Optional[List[Dict[str, Any]]] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_turn(self, role: str, text: str, *, add_generation_prompt: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_generation_prompt(self, role: str = "assistant") -> str:
        raise NotImplementedError

    @abstractmethod
    def render_tool_response(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_messages(
        self,
        *,
        messages: List[Dict[str, str]],
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_tool_calls_from_text(self, text: str) -> Tuple[bool, Optional[List[Optional[ToolCall]]]]:
        raise NotImplementedError


class Qwen3Template(Template):
    def render_generation_prompt(self, role: str = "assistant") -> str:
        return f"{IM_START}{role}\n"

    def render_turn(self, role: str, text: str, *, add_generation_prompt: bool = False) -> str:
        turn = f"{IM_START}{role}\n{text}{IM_END}\n"
        if add_generation_prompt:
            turn += self.render_generation_prompt()
        return turn

    def render_system(
        self,
        text: str,
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        rendered = text
        if tools_manifest:
            rendered = QWEN3_SYSTEM_WITH_TOOLS_TEMPLATE.format(
                system=text,
                tools="\n".join([json.dumps(item, ensure_ascii=False) for item in tools_manifest]),
            )
        return self.render_turn(role="system", text=rendered)

    def render_tool_response(self, text: str) -> str:
        return f"<tool_response>\n{text}\n</tool_response>"

    def render_messages(
        self,
        *,
        messages: List[Dict[str, str]],
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        blocks: List[str] = []
        pending = list(messages)

        if pending and pending[0]["role"] == "system":
            system_msg = pending.pop(0)
            blocks.append(
                self.render_system(
                    system_msg["content"],
                    tools_manifest=tools_manifest,
                )
            )

        for item in pending:
            blocks.append(self.render_turn(item["role"], item.get("content", "")))

        if add_generation_prompt:
            blocks.append(self.render_generation_prompt())

        return "".join(blocks)

    def extract_tool_calls_from_text(self, text: str) -> Tuple[bool, Optional[List[Optional[ToolCall]]]]:
        matches = QWEN3_TOOL_RE.findall(text or "")
        if len(matches) < 1:
            return True, []

        calls: List[Optional[ToolCall]] = []
        for idx, raw in enumerate(matches):
            try:
                obj = json.loads(raw)
                name = obj.get("name")
                if not isinstance(name, str) or not name:
                    calls.append(None)
                    continue

                arguments = obj.get("arguments", {}) or {}
                if not isinstance(arguments, dict):
                    calls.append(None)
                    continue

                calls.append(
                    ToolCall(
                        id=obj.get("id") or f"call_{idx}",
                        name=name,
                        arguments=arguments,
                    )
                )
            except Exception:
                calls.append(None)
        return False, calls


def make_template(name: Optional[str] = None) -> Template:
    if name in (None, "", "qwen3"):
        return Qwen3Template()
    raise ValueError(f"Unknown template '{name}'.")


__all__ = [
    "Template",
    "Qwen3Template",
    "make_template",
]
